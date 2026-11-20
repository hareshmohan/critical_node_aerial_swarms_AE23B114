#!/usr/bin/env python3
"""
Deployment demo: the live criticality score on ONE test case, one time step at a time.

Saved in deployable_output/<case>/: scores_every_step.csv, timing.csv, ranking_overall.csv,
ranking_step*.csv, critical_ranking_step*.csv, forecast_check.csv, fig_score_over_time.png, fig_critical_vs_decoy.png

    python deployable_model.py                         # test case with most agents lost
    python deployable_model.py --case c005_n52_heavy
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb          # lightgbm (and the stacker fit) must come before torch is imported
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")
from swarm_common import load_data, make_snapshots, rows_for, rank_metrics, TwoPartModel, HistoryBuffer
import linear_blend as lb       # stacker inputs and settings (no torch in there)

STACKER_INPUTS = lb.NOW + lb.CONTEXT + lb.MEMORY
SCORE_NAMES = ["lgb", "gs", "stacked", "now", "future", "assign"]
STAGES = ["history_ms", "lgb_ms", "graphsage_ms", "stacker_window_ms"]


def clip01(x):
    return np.clip(x, 0, 1)


# ============================================================================ the stacker
def get_stacker(args, data_dir, out_dir, case):
    """Load the saved stacker, or fit it once and save it. Returns a lightgbm Booster."""
    path = out_dir / "stacker.txt"
    if path.exists() and not args.refit_stacker:
        print("stacker: loaded from", path)
        return lgb.Booster(model_file=str(path))

    t0 = time.time()
    has_val = all((data_dir / d / "val_predictions.parquet").exists() for d in [args.lgb_dir, args.gs_dir])
    if has_val:
        print("stacker: fitting on the validation predictions ...", flush=True)
        table = lb.add_features(lb.merge_models(data_dir, args.lgb_dir, args.gs_dir, "val"))
    else:
        print("stacker: no validation predictions for both models, fitting on the other test cases ...", flush=True)
        table = lb.add_features(lb.merge_models(data_dir, args.lgb_dir, args.gs_dir, "test"))
        table = lb.add_features(table[table["case_id"] != case])

    model = lb.make_stacker().fit(table[STACKER_INPUTS], table["score_true"])
    model.booster_.save_model(str(path))
    print("stacker: %d rows, %d inputs, done in %.0f s" % (len(table), len(STACKER_INPUTS), time.time() - t0))
    return model.booster_


class StackerInputs:
    """Online version of linear_blend.add_features. Keeps a small memory of every agent (how often it was
    seen, its mean scores, its last 20 scores) and adds the context of the current snapshot.
    Only scores of this and earlier steps are used."""

    def __init__(self):
        self.memory = {}

    def reset(self):
        self.memory = {}

    def build(self, agent_ids, p_lgb, p_gs):
        n = len(agent_ids)
        avg = 0.5 * (p_lgb + p_gs)

        n_seen, mean_lgb, mean_gs, mean_avg = (np.zeros(n) for _ in range(4))
        roll5, roll20 = np.zeros(n), np.zeros(n)
        memory = {}                                   # agents that are gone are not carried over
        for i, agent in enumerate(agent_ids):
            old = self.memory.get(agent, {"n": 0, "lgb": 0.0, "gs": 0.0, "avg": 0.0, "last": []})
            last20 = (old["last"] + [float(avg[i])])[-20:]
            now = {"n": old["n"] + 1, "lgb": old["lgb"] + float(p_lgb[i]), "gs": old["gs"] + float(p_gs[i]),
                   "avg": old["avg"] + float(avg[i]), "last": last20}
            memory[agent] = now
            n_seen[i] = now["n"]
            mean_lgb[i] = now["lgb"] / now["n"]
            mean_gs[i] = now["gs"] / now["n"]
            mean_avg[i] = now["avg"] / now["n"]
            roll5[i] = np.mean(last20[-5:])
            roll20[i] = np.mean(last20)
        self.memory = memory

        spread = avg.std(ddof=1) if n > 1 else 0.0
        columns = {
            # now
            "lgb": p_lgb, "gs": p_gs, "avg": avg, "diff": p_gs - p_lgb,
            # context of this snapshot
            "n_alive": np.full(n, float(n)), "snap_mean": np.full(n, avg.mean()), "snap_std": np.full(n, spread),
            "rank_now": rankdata(avg) / n,
            # memory of the agent
            "n_seen": n_seen, "lgb_mean_all": mean_lgb, "gs_mean_all": mean_gs, "avg_mean_all": mean_avg,
            "avg_roll5": roll5, "avg_roll20": roll20, "rank_mean_all": rankdata(mean_avg) / n,
        }
        return np.column_stack([columns[name] for name in STACKER_INPUTS])


class DecisionWindow:
    """Keeps the last `size` stacked scores of every agent.
    A straight line is fitted through them: smoothed score = line value at the latest snapshot,
    slope = slope of the line. Agents with a short history repeat their first score."""

    def __init__(self, size):
        assert size >= 2
        self.size = size
        self.history = {}

    def reset(self):
        self.history = {}

    def update(self, agent_ids, scores):
        rows, history = [], {}
        for agent, score in zip(agent_ids, scores):
            h = (self.history.get(agent, []) + [float(score)])[-self.size:]
            history[agent] = h
            rows.append([h[0]] * (self.size - len(h)) + h)
        self.history = history

        H = np.array(rows)
        t = np.arange(self.size) - (self.size - 1) / 2
        slope = (H - H.mean(axis=1, keepdims=True)) @ t / (t ** 2).sum()
        smoothed = clip01(H.mean(axis=1) + slope * t[-1])
        return smoothed, slope


# ============================================================================ the online pipeline
class OnlineCriticality:
    """One call per time step: features of the alive agents in, all scores out."""

    def __init__(self, lgb_model, gs_model, stacker, radius, window, horizon, w_future, base_feats):
        import torch
        self.torch = torch
        self.lgb_model, self.gs_model, self.stacker = lgb_model, gs_model, stacker
        self.radius, self.horizon, self.w_future = radius, horizon, w_future
        self.history = HistoryBuffer(base_feats)
        self.stack_inputs = StackerInputs()
        self.window = DecisionWindow(window)

    def reset(self):
        self.history.reset()
        self.stack_inputs.reset()
        self.window.reset()

    def graphsage_score(self, X, positions):
        """Radius graph from the positions of this snapshot, then one GraphSAGE pass."""
        torch = self.torch
        with torch.no_grad():
            pos = torch.from_numpy(positions)
            close = torch.cdist(pos, pos) <= self.radius
            close.fill_diagonal_(False)
            dst, src = close.nonzero(as_tuple=True)
            edges = torch.stack([src, dst])
            out = torch.sigmoid(self.gs_model(torch.from_numpy(X), edges).squeeze(-1))
        return clip01(out.numpy().astype(np.float64))

    def step(self, agent_ids, X_base, positions):
        """Returns the six scores, the stacker inputs and the time of each stage (seconds)."""
        t0 = time.perf_counter()
        X = np.hstack([X_base, self.history.update(agent_ids, X_base)])
        t1 = time.perf_counter()
        p_lgb = clip01(self.lgb_model.predict(X, num_threads=1))
        t2 = time.perf_counter()
        p_gs = self.graphsage_score(X, positions)
        t3 = time.perf_counter()
        inputs = self.stack_inputs.build(agent_ids, p_lgb, p_gs)
        stacked = clip01(self.stacker.predict(inputs, num_threads=1))
        smoothed, slope = self.window.update(agent_ids, stacked)
        future = clip01(smoothed + slope * self.horizon)
        assign = (1 - self.w_future) * smoothed + self.w_future * future
        t4 = time.perf_counter()
        scores = (p_lgb, p_gs, stacked, smoothed, future, assign)
        return scores, inputs, (t1 - t0, t2 - t1, t3 - t2, t4 - t3)


def run_stream(online, rows, X_base, positions):
    """Feeds the case snapshot by snapshot, in time order. Returns the scores, the stacker inputs and
    the time per step."""
    agent = rows["agent_id"].to_numpy()
    ts = rows["ts_id"].to_numpy()
    bounds = np.r_[0, np.flatnonzero(np.diff(ts)) + 1, len(ts)]     # row range of every snapshot

    scores = {name: np.zeros(len(rows)) for name in SCORE_NAMES}
    stack_inputs = np.zeros((len(rows), len(STACKER_INPUTS)))
    timing = []
    online.reset()
    for a, b in zip(bounds[:-1], bounds[1:]):
        step_scores, inputs, seconds = online.step(agent[a:b], X_base[a:b], positions[a:b])
        for name, values in zip(SCORE_NAMES, step_scores):
            scores[name][a:b] = values
        stack_inputs[a:b] = inputs
        timing.append([1000 * s for s in seconds] + [b - a])

    timing = pd.DataFrame(timing, columns=STAGES + ["n_agents"])
    timing["total_ms"] = timing[STAGES].sum(axis=1)
    return scores, stack_inputs, timing


def future_truth(rows, horizon):
    """Data score and label of the same agent `horizon` snapshots later (NaN if the agent is gone)."""
    k = rows.groupby("case_id")["ts_id"].rank(method="dense").astype(int)
    keys = pd.DataFrame({"case_id": rows["case_id"], "agent_id": rows["agent_id"], "k": k + horizon})
    later = pd.DataFrame({"case_id": rows["case_id"], "agent_id": rows["agent_id"], "k": k,
                          "score_later": rows["score_true"], "crit_later": rows["is_critical"]})
    merged = keys.merge(later, on=["case_id", "agent_id", "k"], how="left")
    return merged["score_later"].to_numpy(), merged["crit_later"].to_numpy()


# ============================================================================ setup
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--lgb-dir", default="results_score_lightgbm")
    ap.add_argument("--gs-dir", default="results_score_graphsage_r25",
                    help="GraphSAGE result folder (the range-25 run has validation predictions for the stacker)")
    ap.add_argument("--out", default="deployable_output")
    ap.add_argument("--case", default="", help="test case to run (default: test case with most agents lost)")
    ap.add_argument("--window", type=int, default=5, help="decision window: latest snapshots kept per agent")
    ap.add_argument("--horizon", type=int, default=5, help="how many snapshots ahead the future score looks")
    ap.add_argument("--w-future", type=float, default=0.5, help="weight of the future score in the assignment score")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="assignment score at or above this is classified critical "
                         "(about the F2 threshold of the stacker on validation, see stack_blend_metrics.csv)")
    ap.add_argument("--refit-stacker", action="store_true")
    return ap.parse_args()


def pick_case(data_dir):
    """The test case in which the most agents are lost."""
    meta = pd.read_parquet(data_dir / "processed" / "test_tree.parquet",
                           columns=["case_id", "strat_num_agents", "n_alive"])
    lost = (meta["strat_num_agents"] - meta["n_alive"]).groupby(meta["case_id"]).max()
    return lost.sort_values(ascending=False).index[0]


def load_models(args, data_dir):
    """The two trained score models. torch is imported here, after lightgbm."""
    lgb_cfg = json.load(open(data_dir / args.lgb_dir / "run_config.json"))
    gs_cfg = json.load(open(data_dir / args.gs_dir / "run_config.json"))
    assert lgb_cfg["features"] == gs_cfg["features"], "the two models must use the same features"

    lgb_model = TwoPartModel.load(data_dir / args.lgb_dir)

    import torch
    from graphsage_critical_node import build_model
    torch.set_num_threads(1)
    gs_model = build_model(gs_cfg["aggregator"], len(lgb_cfg["features"]), gs_cfg["hidden"],
                           gs_cfg["layers"], gs_cfg["dropout"])
    gs_model.load_state_dict(torch.load(data_dir / args.gs_dir / "graphsage_score.pt"))
    gs_model.eval()
    return lgb_model, gs_model, lgb_cfg, gs_cfg


def load_case(data_dir, case, lgb_cfg):
    """Only the chosen test case is read. Returns its rows, features and positions."""
    base_feats = lgb_cfg["base_features"]
    df, X_all, names = load_data(data_dir, base_feats, 0, splits=["test"], history=lgb_cfg["history"],
                                 only_cases={"test": [case]})
    assert names == lgb_cfg["features"]
    snaps, offs = make_snapshots(df)
    _, idx = rows_for(snaps, offs, "test", 1)

    rows = df.iloc[idx].reset_index(drop=True)
    assert (rows["case_id"] == case).all(), "case not in the test split: " + case
    assert rows.groupby("agent_id")["step"].apply(lambda s: s.is_monotonic_increasing).all(), \
        "rows must be sorted by step"
    positions = df[["rel_x", "rel_y", "rel_z"]].to_numpy(np.float32)[idx]
    return rows, X_all[idx], positions


# ============================================================================ checks and reports
def check_against_offline(rows, stream, stack_inputs, lgb_model, X):
    """The live computation must give the same numbers as the offline one."""
    batch_lgb = clip01(lgb_model.predict(X, num_threads=1))
    diff = np.abs(stream["lgb"] - batch_lgb)
    assert diff.mean() < 1e-3 and (diff > 1e-2).mean() < 0.01, "live and batch LightGBM scores differ"

    keys = ["case_id", "step", "agent_id"]
    offline = lb.add_features(pd.DataFrame({"case_id": rows["case_id"], "step": rows["step"],
                                            "agent_id": rows["agent_id"],
                                            "lgb": stream["lgb"], "gs": stream["gs"]}))
    offline = rows[keys].merge(offline, on=keys, how="left")
    assert np.abs(offline[STACKER_INPUTS].to_numpy() - stack_inputs).max() < 1e-6, \
        "online stacker inputs differ from linear_blend.py"
    print("check passed: live history features, LightGBM scores and stacker inputs equal the offline computation")


def report_timing(timing, case_dir):
    cols = STAGES + ["total_ms"]
    table = pd.DataFrame({"mean_ms": timing[cols].mean(), "p95_ms": timing[cols].quantile(0.95),
                          "max_ms": timing[cols].max()})
    table.index = ["history features", "LightGBM scoring", "GraphSAGE scoring (incl. graph building)",
                   "stacker + window + future", "total per time step"]
    table.to_csv(case_dir / "timing.csv")
    print("\n=== time per time step (about %.0f agents alive, one snapshot at a time) ===" % timing["n_agents"].mean())
    print(table.round(2))


def report_rankings(c, case_dir):
    """Ranking at the first, middle and last step, and over the whole case. Returns the overall table."""
    steps = np.sort(c["step"].unique())
    for step in [steps[0], steps[len(steps) // 2], steps[-1]]:
        r = c[c["step"] == step].sort_values("assign", ascending=False).reset_index(drop=True)
        r.index = r.index + 1
        r = r[["agent_id", "assign", "now", "future", "score_true", "is_critical"]]
        r.columns = ["agent", "our score", "smoothed", "future", "data score", "critical in data"]
        r.rename_axis("rank").to_csv(case_dir / ("ranking_step%d.csv" % step))
        print("\nranking at step %d (%d agents alive), top 10:" % (step, len(r)))
        print(r.head(10).round(3))

    overall = c.groupby("agent_id").agg(mean_score=("assign", "mean"), steps_alive=("step", "size"),
                                        mean_data_score=("score_true", "mean"),
                                        share_critical=("is_critical", "mean"))
    overall = overall.sort_values("mean_score", ascending=False)
    overall.to_csv(case_dir / "ranking_overall.csv")
    print("\nagents ranked by their mean score over the case, top 10:")
    print(overall.head(10).round(3))
    return overall


def report_critical_ranking(c, threshold, case_dir):
    """At the first, middle and last step: every agent classified as critical (assignment score >= threshold),
    ranked from most to least critical. Saved as critical_ranking_step*.csv and printed in full."""
    steps = np.sort(c["step"].unique())
    for step in [steps[0], steps[len(steps) // 2], steps[-1]]:
        snap = c[c["step"] == step]
        r = snap[snap["assign"] >= threshold].sort_values("assign", ascending=False).reset_index(drop=True)
        r.index = r.index + 1
        r = r[["agent_id", "assign", "now", "future", "score_true", "is_critical"]]
        r.columns = ["agent", "our score", "smoothed", "future", "data score", "critical in data"]
        r.rename_axis("rank").to_csv(case_dir / ("critical_ranking_step%d.csv" % step))
        n_right = int(r["critical in data"].sum())
        print("\ncritical agents at step %d: %d of %d alive classified critical (score >= %.3f), %d of them critical in the data"
              % (step, len(r), len(snap), threshold, n_right))
        print(r.round(3).to_string())


def report_quality(c):
    ts, y, truth = c["ts_id"].to_numpy(), c["is_critical"].to_numpy(), c["score_true"].to_numpy()
    rows = {}
    for name in ["lgb", "gs", "stacked", "now", "assign"]:
        rm = rank_metrics(ts, y, c[name].to_numpy(), truth)
        rows[name] = {"MAE vs data score": np.abs(c[name] - truth).mean(),
                      "PR-AUC": average_precision_score(y, c[name]),
                      "R-precision": rm["r_precision"], "top-1 hit": rm["top1_hit"], "precision@5": rm["p_at_5"]}
    print("\n=== quality in this case ===")
    print(pd.DataFrame(rows).T.rename(index={"lgb": "LightGBM", "gs": "GraphSAGE", "stacked": "stacked score",
                                              "now": "smoothed (window)", "assign": "assignment score"}).round(3))


def report_forecast(c, horizon, case_dir):
    """Does the score at step t match the data `horizon` snapshots later?"""
    ts = c["ts_id"].to_numpy()
    score_later, crit_later = future_truth(c, horizon)
    ok = ~np.isnan(score_later)
    rows = []
    for name, label in [("score_true", "data score at t (reference)"), ("stacked", "stacked score at t"),
                        ("now", "smoothed score at t"), ("assign", "assignment score at t")]:
        p = c[name].to_numpy()[ok]
        rm = rank_metrics(ts[ok], crit_later[ok].astype(int), p, score_later[ok])
        rows.append({"score": label, "MAE vs data score at t+H": np.abs(p - score_later[ok]).mean(),
                     "PR-AUC vs critical at t+H": average_precision_score(crit_later[ok].astype(int), p),
                     "R-precision": rm["r_precision"]})
    table = pd.DataFrame(rows).set_index("score")
    table.to_csv(case_dir / "forecast_check.csv")
    print("\n=== how well does the score at step t match the data %d snapshots later ===" % horizon)
    print(table.round(4))


# ============================================================================ figures
def plot_top_agents(c, overall, case, topk, case_dir):
    """Online score (solid) and data score (dashed) of the top agents; grey lines mark agent losses."""
    n_alive = c.groupby("step")["agent_id"].size().diff()
    loss_steps = n_alive[n_alive < 0].index

    fig, ax = plt.subplots(figsize=(11, 5.5))
    colors = plt.cm.tab10.colors
    for j, agent in enumerate(overall.index[:topk]):
        g = c[c["agent_id"] == agent]
        ax.plot(g["step"], g["assign"], color=colors[j], lw=1.6, label="agent %d" % agent)
        ax.plot(g["step"], g["score_true"], color=colors[j], lw=0.8, ls="--", alpha=0.7)
    for step in loss_steps:
        ax.axvline(step, color="grey", lw=0.4, alpha=0.5)
    ax.set_xlabel("time step")
    ax.set_ylabel("criticality score")
    ax.set_title("Case %s: our online score (solid) vs data score (dashed), top %d agents; grey lines = agent lost"
                 % (case, topk))
    ax.legend(ncol=topk, fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.savefig(case_dir / "fig_score_over_time.png", dpi=150)
    plt.close()


def plot_critical_vs_decoy(c, case, case_dir):
    """Mean score of critical agents and decoys per step, with a 10-90% band."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, color in [(1, "tab:red"), (0, "tab:blue")]:
        g = c[c["is_critical"] == label].groupby("step")["assign"]
        mean = g.mean()
        name = "critical" if label else "decoy"
        ax.plot(mean.index, mean.values, color=color, label=name + " agents (mean, band = 10-90%)")
        ax.fill_between(mean.index, g.quantile(0.1).values, g.quantile(0.9).values, color=color, alpha=0.15)
    ax.set_xlabel("time step")
    ax.set_ylabel("our criticality score")
    ax.set_title("Case %s: score of critical agents and decoys over time" % case)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(case_dir / "fig_critical_vs_decoy.png", dpi=150)
    plt.close()


# ============================================================================ main
def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = data_dir / args.out
    out_dir.mkdir(exist_ok=True)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    t_start = time.time()

    # 1. which case, and the stacker (both before torch is imported)
    case = args.case or pick_case(data_dir)
    stacker = get_stacker(args, data_dir, out_dir, case)

    # 2. the two trained score models
    lgb_model, gs_model, lgb_cfg, gs_cfg = load_models(args, data_dir)
    radius = gs_cfg["range"]
    print("features: %d | GraphSAGE: %s, range %g | window %d | horizon %d | weight of future score %.2f" %
          (len(lgb_cfg["features"]), gs_cfg["aggregator"], radius, args.window, args.horizon, args.w_future))

    # 3. the case
    c, X, positions = load_case(data_dir, case, lgb_cfg)
    print("\ncase %s: %d time steps, %d agents, %d lost at the end, %d critical agents (loaded in %.0f s)" %
          (case, c["step"].nunique(), c["agent_id"].nunique(), int(c["n_lost"].max()),
           int(c["is_critical"].groupby(c["step"]).sum().iloc[0]), time.time() - t_start))
    case_dir = out_dir / case
    case_dir.mkdir(exist_ok=True)

    # 4. live run, then check it against the offline computation
    base_feats = lgb_cfg["base_features"]
    online = OnlineCriticality(lgb_model, gs_model, stacker, radius, args.window, args.horizon,
                               args.w_future, base_feats)
    stream, stack_inputs, timing = run_stream(online, c, X[:, :len(base_feats)], positions)
    for name, values in stream.items():
        c[name] = values
    check_against_offline(c, stream, stack_inputs, lgb_model, X)

    c[["case_id", "step", "agent_id", "is_critical", "score_true"] + SCORE_NAMES] \
        .to_csv(case_dir / "scores_every_step.csv", index=False)

    # 5. reports
    report_timing(timing, case_dir)
    overall = report_rankings(c, case_dir)
    report_critical_ranking(c, args.threshold, case_dir)
    report_quality(c)
    report_forecast(c, args.horizon, case_dir)

    # 6. figures
    plot_top_agents(c, overall, case, args.topk, case_dir)
    plot_critical_vs_decoy(c, case, case_dir)
    print("\nsaved to", case_dir, "| total run time %.0f s" % (time.time() - t_start))


if __name__ == "__main__":
    main()
