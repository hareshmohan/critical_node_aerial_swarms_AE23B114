"""
Shared code for the LightGBM, GraphSAGE and stacking scripts.

Input: position + kinematic features from preprocessing.ipynb (processed/*_nn.parquet), plus history
features of every agent's own past. No neighbour / graph columns, environment, attrition or aux_ columns
are used as inputs.
Target: aux_criticality_score (0 to 1).

History features use only past and present rows, so HistoryBuffer (one snapshot at a time) gives the
same numbers as history_features (whole table, used for training).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

SPLITS = ["train", "val", "test"]
KEY = ["case_id", "step", "agent_id"]

# Dubins-like state: position, speed, heading, turn rate, climb angle, acceleration, saturation flags
FEATURE_GROUPS = {
    "raw": ["rel_x", "rel_y", "rel_z", "dist_to_centroid", "rank_dist_to_centroid", "dist_to_target",
            "rank_dist_to_target", "speed", "flight_path_angle", "accel", "heading_rate"],
    "local": ["heading_sin", "heading_cos", "rel_azimuth_sin", "rel_azimuth_cos", "rel_elevation",
              "radial_speed", "turn_saturated", "pitch_saturated"],
    "context": ["swarm_radius", "swarm_mean_dist", "dist_centroid_rel_radius", "rel_speed_to_swarm",
                "heading_dev_abs", "speed_z", "n_alive"],
    # not used by default
    "attrition": ["attrition_level", "alive_frac"],
    "environment": ["n_obstacles", "obstacle_density", "mean_obstacle_size", "clutter_index", "obstacle_clustering"],
}
TARGET_COLS = ["dist_to_target", "rank_dist_to_target"]


def get_features(groups, no_target=False):
    feats = []
    for g in groups:
        for f in FEATURE_GROUPS[g]:
            if no_target and f in TARGET_COLS:
                continue
            feats.append(f)
    return feats


def load_data(data_dir, feats, max_cases=0, splits=SPLITS, history=True, only_cases=None):
    """Reads the preprocessed files. Returns df (one row per agent per step), X (scaled features) and the
    feature names. history=True adds the history features after the base features.
    only_cases: optional {split: [case ids]} to load only these cases of a split."""
    pdir = Path(data_dir) / "processed"
    meta_cols = KEY + ["ts_id", "is_critical", "aux_criticality_score", "strat_num_agents", "strat_attrition",
                       "strat_n_obstacles", "strat_obstacle_density", "rel_x", "rel_y", "rel_z", "n_alive"]
    metas = []
    xs = []
    for sp in splits:
        t = pd.read_parquet(pdir / (sp + "_tree.parquet"), columns=meta_cols)     # unscaled: meta + positions
        n = pd.read_parquet(pdir / (sp + "_nn.parquet"), columns=KEY + feats)      # scaled features
        for c in KEY:
            assert (n[c].to_numpy() == t[c].to_numpy()).all(), "tree and nn rows are not aligned"
        t["split"] = sp
        x = n[feats]
        if only_cases is not None and sp in only_cases:
            keep = t["case_id"].isin(only_cases[sp]).to_numpy()
            t = t[keep]
            x = x[keep]
        if max_cases > 0:
            keep = t["case_id"].isin(t["case_id"].unique()[:max_cases]).to_numpy()
            t = t[keep]
            x = x[keep]
        metas.append(t.reset_index(drop=True))
        xs.append(x.reset_index(drop=True))
    df = pd.concat(metas, ignore_index=True)
    X = pd.concat(xs, ignore_index=True).to_numpy(np.float32)
    assert np.isfinite(X).all(), "non-finite feature values"

    df["is_critical"] = df["is_critical"].astype(int)
    df = df.rename(columns={"aux_criticality_score": "score_true", "strat_num_agents": "num_agents",
                            "strat_attrition": "attrition", "strat_n_obstacles": "n_obstacles",
                            "strat_obstacle_density": "obstacle_density"})
    df["n_lost"] = df["num_agents"] - df["n_alive"]
    df["ts_id"] = df.groupby("ts_id", sort=False).ngroup()      # new snapshot number
    first_rows = (df["ts_id"].diff().fillna(1) != 0).sum()
    assert first_rows == df["ts_id"].nunique(), "snapshot rows are not contiguous"

    # no case may be in two splits
    for a in splits:
        for b in splits:
            if a < b:
                assert set(df.loc[df.split == a, "case_id"]).isdisjoint(set(df.loc[df.split == b, "case_id"]))

    names = list(feats)
    if history:
        X = np.hstack([X, history_features(df, X, feats)])
        names = names + history_names(feats)
    return df, X, names


# ---------------------------------------------------------------- history features (own past only)
HIST_COLS = ["dist_to_centroid", "rank_dist_to_centroid", "dist_to_target", "rank_dist_to_target", "speed", "accel",
             "heading_rate", "flight_path_angle", "radial_speed", "rel_speed_to_swarm", "heading_dev_abs", "speed_z",
             "dist_centroid_rel_radius", "rel_elevation", "turn_saturated", "pitch_saturated", "heading_sin",
             "heading_cos"]
ROLL_COLS = ["accel", "heading_rate", "flight_path_angle", "speed", "rel_speed_to_swarm", "radial_speed",
             "heading_dev_abs", "speed_z"]
ROLL_WINDOWS = [10, 30]


def history_columns(base_feats):
    """Positions (inside base_feats) of the columns used for the history features."""
    hist = [base_feats.index(c) for c in HIST_COLS if c in base_feats]
    roll = [base_feats.index(c) for c in ROLL_COLS if c in base_feats]
    return hist, roll


def history_names(base_feats):
    hist = [c for c in HIST_COLS if c in base_feats]
    roll = [c for c in ROLL_COLS if c in base_feats]
    names = ["mean_" + c for c in hist] + ["std_" + c for c in hist]
    for w in ROLL_WINDOWS:
        names = names + ["rmean%d_%s" % (w, c) for c in roll] + ["rstd%d_%s" % (w, c) for c in roll]
    return names


def history_features(df, X_base, base_feats):
    """History features for every row. Mean / std of an agent's own features since the start of its case,
    and over the last 10 and 30 snapshots (past and present rows only)."""
    hist, roll = history_columns(base_feats)
    case_codes = pd.factorize(df["case_id"])[0].astype(np.int64)
    agent = df["agent_id"].to_numpy().astype(np.int64)
    assert agent.max() < 1000
    order = np.lexsort((df["ts_id"].to_numpy(), agent, case_codes))       # one agent's rows together, in time order
    gid = case_codes[order] * 1000 + agent[order]
    group_start = np.r_[0, np.flatnonzero(np.diff(gid)) + 1]
    gstart = np.repeat(group_start, np.diff(np.r_[group_start, len(gid)]))
    pos = np.arange(len(gid)) - gstart                                    # 0, 1, 2 ... inside a group

    def cum(a):                                                           # cumulative sum per group
        c = np.cumsum(a, axis=0)
        before = np.where((gstart > 0)[:, None], c[np.maximum(gstart - 1, 0)], 0.0)
        return c - before

    def mean_std(cs, cs2, window):
        if window is None:
            cnt = (pos + 1)[:, None]
            m = cs / cnt
            v = cs2 / cnt - m ** 2
        else:
            ok = (pos >= window)[:, None]
            idx = np.maximum(np.arange(len(gid)) - window, 0)
            cnt = np.minimum(pos + 1, window)[:, None]
            m = (cs - np.where(ok, cs[idx], 0.0)) / cnt
            v = (cs2 - np.where(ok, cs2[idx], 0.0)) / cnt - m ** 2
        return m, np.sqrt(np.maximum(v, 0))

    out = np.zeros((len(df), len(history_names(base_feats))), np.float32)
    col = 0
    blocks = []
    xs = X_base[order][:, hist].astype(np.float64)
    m, sd = mean_std(cum(xs), cum(xs ** 2), None)
    blocks += [m, sd]
    xs = X_base[order][:, roll].astype(np.float64)
    cs, cs2 = cum(xs), cum(xs ** 2)
    for w in ROLL_WINDOWS:
        m, sd = mean_std(cs, cs2, w)
        blocks += [m, sd]
    for b in blocks:
        out[order, col:col + b.shape[1]] = b
        col += b.shape[1]
    return out


class HistoryBuffer:
    """Same history features, one snapshot at a time (live use). Call update() once per snapshot, in time order,
    with the base features of the alive agents. Call reset() at the start of every case."""
    def __init__(self, base_feats):
        self.hist, self.roll = history_columns(base_feats)
        self.state = {}

    def reset(self):
        self.state = {}

    def update(self, agent_ids, X_base):
        new_state = {}
        rows = []
        for j in range(len(agent_ids)):
            agent = agent_ids[j]
            x = X_base[j].astype(np.float64)
            st = self.state.get(agent)
            if st is None:
                st = {"n": 0, "s1": np.zeros(len(self.hist)), "s2": np.zeros(len(self.hist)), "recent": []}
            h = x[self.hist]
            st["n"] += 1
            st["s1"] = st["s1"] + h
            st["s2"] = st["s2"] + h * h
            st["recent"] = (st["recent"] + [x[self.roll]])[-max(ROLL_WINDOWS):]
            new_state[agent] = st
            mean = st["s1"] / st["n"]
            parts = [mean, np.sqrt(np.maximum(st["s2"] / st["n"] - mean ** 2, 0))]
            recent = np.array(st["recent"])
            for w in ROLL_WINDOWS:
                last = recent[-w:]
                parts += [last.mean(axis=0), last.std(axis=0)]
            rows.append(np.concatenate(parts))
        self.state = new_state                         # lost agents are dropped
        return np.array(rows, dtype=np.float32)


def make_snapshots(df):
    """One row per snapshot, and the start row of every snapshot (offs)."""
    counts = df.groupby("ts_id", sort=True).size().to_numpy()
    offs = np.r_[0, np.cumsum(counts)]
    snaps = df.groupby("ts_id", sort=True).agg(case_id=("case_id", "first"), step=("step", "first"),
                                               split=("split", "first")).reset_index()
    snaps["k"] = snaps.groupby("case_id").cumcount()
    return snaps, offs


def rows_for(snaps, offs, split, stride):
    """Snapshots of one split (every stride-th) and their row numbers."""
    s = snaps[(snaps.split == split) & (snaps.k % stride == 0)]
    idx = []
    for ts in s.ts_id:
        idx.append(np.arange(offs[ts], offs[ts + 1]))
    return s, np.concatenate(idx)


# ---------------------------------------------------------------- metrics
def find_threshold(y, p):
    """Score threshold with the best F2 (recall counts more). Use validation data."""
    precision, recall, thresholds = precision_recall_curve(y, p)
    precision = precision[:-1]
    recall = recall[:-1]
    f2 = 5 * precision * recall / np.maximum(4 * precision + recall, 1e-12)
    return float(thresholds[np.argmax(f2)])


def score_metrics(true_score, p):
    """How close the predicted score is to the data score."""
    err = p - true_score
    ss_res = (err ** 2).sum()
    ss_tot = ((true_score - true_score.mean()) ** 2).sum()
    return {"mae": np.abs(err).mean(), "rmse": np.sqrt((err ** 2).mean()), "r2": 1 - ss_res / ss_tot,
            "spearman": pd.Series(true_score).corr(pd.Series(p), method="spearman")}


def class_metrics(y, p, thr):
    """Critical / not critical from the score, using the threshold thr."""
    pred = p >= thr
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    tn = int((~pred & (y == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    result = {"accuracy": (tp + tn) / len(y), "precision": precision, "recall": recall,
              "f1": 2 * precision * recall / max(precision + recall, 1e-12),
              "f2": 5 * precision * recall / max(4 * precision + recall, 1e-12), "threshold": thr}
    if len(np.unique(y)) > 1:
        result["pr_auc"] = average_precision_score(y, p)
        result["roc_auc"] = roc_auc_score(y, p)
    else:
        result["pr_auc"] = np.nan
        result["roc_auc"] = np.nan
    return result


def rank_metrics(ts, y, p, true_score=None, k=5):
    """Ranking quality per snapshot (agents ranked by p), averaged over snapshots.
    r_precision: precision in the top K, K = number of critical agents. top1_hit: best ranked agent is critical.
    top5_overlap: share of the true top 5 (by data score) found in the predicted top 5."""
    cuts = np.flatnonzero(np.diff(ts)) + 1
    ys = np.split(y, cuts)
    ps = np.split(p, cuts)
    ts_list = None
    if true_score is not None:
        ts_list = np.split(true_score, cuts)
    r_prec = []
    avg_prec = []
    top1 = []
    p_at_k = []
    overlap = []
    for i in range(len(ys)):
        n_crit = int(ys[i].sum())
        if n_crit == 0:
            continue
        order = np.argsort(-ps[i], kind="stable")
        ranked = ys[i][order]
        r_prec.append(ranked[:n_crit].mean())
        hits = np.cumsum(ranked)
        avg_prec.append((hits / np.arange(1, len(ranked) + 1))[ranked == 1].mean())
        top1.append(ranked[0])
        p_at_k.append(ranked[:k].mean())
        if ts_list is not None:
            kk = min(k, len(ranked))
            true_top = set(np.argsort(-ts_list[i], kind="stable")[:kk])
            pred_top = set(order[:kk])
            overlap.append(len(true_top & pred_top) / kk)
    result = {"r_precision": np.mean(r_prec), "map": np.mean(avg_prec), "top1_hit": np.mean(top1),
              "p_at_5": np.mean(p_at_k), "n_snapshots": len(r_prec)}
    if ts_list is not None:
        result["top5_overlap"] = np.mean(overlap)
    return result


def all_metrics(df_rows, p, thr):
    """Everything for one set of rows (df_rows) and predicted scores p."""
    y = df_rows["is_critical"].to_numpy()
    true_score = df_rows["score_true"].to_numpy()
    ts = df_rows["ts_id"].to_numpy()
    result = score_metrics(true_score, p)
    result.update(class_metrics(y, p, thr))
    result.update(rank_metrics(ts, y, p, true_score))
    result["prevalence"] = y.mean()
    return result


# ---------------------------------------------------------------- subsets (size, obstacles, attrition)
def obstacle_cuts(df):
    """Cut points for few / mid / many obstacles (thirds of the training cases)."""
    train_cases = df[df.split == "train"].drop_duplicates("case_id")
    return train_cases["n_obstacles"].quantile([1 / 3, 2 / 3]).to_numpy()


def subset_labels(df, cuts):
    """For every row: attrition level, swarm size group, obstacle group, number of agents lost."""
    labels = {}
    labels["attrition"] = df["attrition"].astype(str).to_numpy()
    labels["swarm size"] = pd.cut(df["num_agents"], [0, 30, 52, 100],
                                  labels=["18-30", "37-52", "60-75"]).astype(str).to_numpy()
    n_obs = df["n_obstacles"].to_numpy()
    obs = np.where(n_obs <= cuts[0], "few obstacles", np.where(n_obs <= cuts[1], "mid obstacles", "many obstacles"))
    labels["obstacles"] = obs
    labels["agents lost"] = pd.cut(df["n_lost"], [-1, 0, 5, 15, 100],
                                   labels=["0", "1-5", "6-15", "16+"]).astype(str).to_numpy()
    return labels


def subset_table(df_rows, p, thr, cuts, model_name):
    """Metrics of one model for every subset of the rows."""
    labels = subset_labels(df_rows, cuts)
    out = []
    for kind in labels:
        for group in np.unique(labels[kind]):
            mask = labels[kind] == group
            d = df_rows[mask]
            if mask.sum() < 200 or d["is_critical"].nunique() < 2:
                continue
            m = all_metrics(d, p[mask], thr)
            m.update(model=model_name, subset=kind + ": " + group, n_rows=int(mask.sum()),
                     n_cases=d["case_id"].nunique())
            out.append(m)
    return pd.DataFrame(out)


# ---------------------------------------------------------------- LightGBM two-part score model
class TwoPartModel:
    """score = p * s1 + (1 - p) * s0
    p  = probability of being critical (LightGBM classifier on is_critical)
    s1 = LightGBM regressor of the data score, trained on critical agents only
    s0 = LightGBM regressor of the data score, trained on decoy agents only"""
    def __init__(self):
        self.clf = None
        self.reg1 = None
        self.reg0 = None

    def fit(self, X_tr, y_tr, s_tr, X_va, y_va, s_va, seed=0, n_estimators=3000, patience=80):
        import lightgbm as lgb
        common = dict(n_estimators=n_estimators, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0, n_jobs=-1,
                      random_state=seed, verbose=-1)
        clf = lgb.LGBMClassifier(**common)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
                callbacks=[lgb.early_stopping(patience, verbose=False)])
        regs = {}
        for c in [0, 1]:
            reg = lgb.LGBMRegressor(objective="regression", **common)
            reg.fit(X_tr[y_tr == c], s_tr[y_tr == c], eval_set=[(X_va[y_va == c], s_va[y_va == c])],
                    callbacks=[lgb.early_stopping(patience, verbose=False)])
            regs[c] = reg
        self.clf, self.reg0, self.reg1 = clf.booster_, regs[0].booster_, regs[1].booster_
        return {"clf_iters": clf.best_iteration_, "reg0_iters": regs[0].best_iteration_, "reg1_iters": regs[1].best_iteration_}

    def predict(self, X, num_threads=-1):
        p = self.clf.predict(X, num_threads=num_threads)
        s1 = np.clip(self.reg1.predict(X, num_threads=num_threads), 0, 1)
        s0 = np.clip(self.reg0.predict(X, num_threads=num_threads), 0, 1)
        return p * s1 + (1 - p) * s0

    def gain_importance(self):
        """Average share of gain of the three models, one value per feature."""
        total = 0
        for b in [self.clf, self.reg1, self.reg0]:
            g = b.feature_importance("gain")
            total = total + g / g.sum()
        return total / 3

    def save(self, folder):
        folder = Path(folder)
        self.clf.save_model(str(folder / "lgb_classifier.txt"))
        self.reg1.save_model(str(folder / "lgb_score_critical.txt"))
        self.reg0.save_model(str(folder / "lgb_score_decoy.txt"))

    @classmethod
    def load(cls, folder):
        import lightgbm as lgb
        folder = Path(folder)
        m = cls()
        m.clf = lgb.Booster(model_file=str(folder / "lgb_classifier.txt"))
        m.reg1 = lgb.Booster(model_file=str(folder / "lgb_score_critical.txt"))
        m.reg0 = lgb.Booster(model_file=str(folder / "lgb_score_decoy.txt"))
        return m
