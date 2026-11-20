"""Stacker: combines the LightGBM and GraphSAGE scores

A small LightGBM learns how to mix the two scores, using the scores, the snapshot context (number of
agents, rank of the agent) and the agent's own score memory (running averages since the start of the
case). Everything uses past scores only, so it can run live.

Fit data: validation predictions of both models. Test is only for evaluation. If the validation files
are missing, the test cases are split in two halves (by case): fit on one, evaluate on the other.

Run after ensemble_kinematic_baselines.py and graphsage_critical_node.py:
    python linear_blend.py
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LinearRegression
from swarm_common import all_metrics, find_threshold

KEY = ["case_id", "step", "agent_id"]

NOW = ["lgb", "gs", "avg", "diff"]
CONTEXT = ["n_alive", "snap_mean", "snap_std", "rank_now"]
MEMORY = ["n_seen", "lgb_mean_all", "gs_mean_all", "avg_mean_all", "avg_roll5", "avg_roll20", "rank_mean_all"]


def merge_models(data_dir, lgb_dir, gs_dir, split):
    a = pd.read_parquet(Path(data_dir) / lgb_dir / (split + "_predictions.parquet"))
    b = pd.read_parquet(Path(data_dir) / gs_dir / (split + "_predictions.parquet"))
    df = a.merge(b[KEY + ["score_gs"]], on=KEY)
    assert len(df) == len(a), "rows of the two models do not match"
    return df.rename(columns={"score_lgb": "lgb", "score_gs": "gs"})


def add_features(df):
    """Adds the stacker inputs. Each row only uses scores of the same step or earlier steps."""
    df = df.sort_values(["case_id", "step", "agent_id"]).reset_index(drop=True)
    df["ts_id"] = df.groupby(["case_id", "step"], sort=False).ngroup()
    df["avg"] = 0.5 * (df["lgb"] + df["gs"])
    df["diff"] = df["gs"] - df["lgb"]

    # context of the snapshot
    snap = df.groupby("ts_id")["avg"]
    df["n_alive"] = snap.transform("size")
    df["snap_mean"] = snap.transform("mean")
    df["snap_std"] = snap.transform("std").fillna(0)
    df["rank_now"] = snap.rank(pct=True)

    # memory of every agent since the start of the case
    g = df.groupby(["case_id", "agent_id"], sort=False)
    df["n_seen"] = g.cumcount() + 1
    for c in ["lgb", "gs", "avg"]:
        df[c + "_mean_all"] = g[c].cumsum() / df["n_seen"]
    df["avg_roll5"] = g["avg"].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df["avg_roll20"] = g["avg"].transform(lambda x: x.rolling(20, min_periods=1).mean())
    df["rank_mean_all"] = df.groupby("ts_id")["avg_mean_all"].rank(pct=True)
    return df


def make_stacker(seed=0):
    return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=200,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=seed, verbose=-1, n_jobs=4)


def fit_predict(model_name, feats, fit_df, eval_df):
    """Fits on fit_df, returns (scores for fit_df from 4-fold by case, scores for eval_df).
    The fit_df scores are out-of-fold, so a threshold picked on them is honest."""
    def one(tr, te_df):
        if model_name == "linear":
            m = LinearRegression().fit(tr[feats], tr["score_true"])
        else:
            m = make_stacker().fit(tr[feats], tr["score_true"])
        return np.clip(m.predict(te_df[feats]), 0, 1)

    cases = fit_df["case_id"].unique()
    fold = fit_df["case_id"].map({c: i % 4 for i, c in enumerate(cases)}).to_numpy()
    oof = np.zeros(len(fit_df))
    for f in range(4):
        oof[fold == f] = one(fit_df[fold != f], fit_df[fold == f])
    return oof, one(fit_df, eval_df)


def run(fit_df, eval_df):
    """One table row per method, evaluated on eval_df."""
    methods = [("LightGBM", None, ["lgb"]), ("GraphSAGE", None, ["gs"]),
               ("Linear blend of both", "linear", ["lgb", "gs"]),
               ("Stacker: scores + snapshot context", "stack", NOW + CONTEXT),
               ("Stacker + long memory (final)", "stack", NOW + CONTEXT + MEMORY)]
    rows = {}
    for name, kind, feats in methods:
        if kind is None:
            p_fit = fit_df[feats[0]].to_numpy()
            p_eval = eval_df[feats[0]].to_numpy()
        else:
            p_fit, p_eval = fit_predict(kind, feats, fit_df, eval_df)
        thr = find_threshold(fit_df["is_critical"].to_numpy(), p_fit)
        rows[name] = all_metrics(eval_df, p_eval, thr)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--lgb-dir", default="results_score_lightgbm")
    ap.add_argument("--gs-dir", default="results_score_graphsage")
    ap.add_argument("--out", default="stack_blend_metrics.csv", help="csv file for the metrics")
    a = ap.parse_args()

    test = add_features(merge_models(a.data_dir, a.lgb_dir, a.gs_dir, "test"))
    have_val = all((Path(a.data_dir) / d / "val_predictions.parquet").exists() for d in [a.lgb_dir, a.gs_dir])

    if have_val:
        print("fitting on validation predictions, evaluating on test")
        val = add_features(merge_models(a.data_dir, a.lgb_dir, a.gs_dir, "val"))
        results = [run(val, test)]
    else:
        print("no validation predictions found: fitting on half of the test cases, evaluating on the other half (2 folds)")
        cases = test["case_id"].unique()
        fold = test["case_id"].map({c: i % 2 for i, c in enumerate(cases)}).to_numpy()
        results = [run(test[fold != f], test[fold == f]) for f in [0, 1]]

    names = list(results[0])
    table = pd.DataFrame({n: {k: np.mean([r[n][k] for r in results]) for k in results[0][n]} for n in names}).T
    table.index.name = "method"
    keep = ["mae", "rmse", "r2", "spearman", "accuracy", "precision", "recall", "f1", "pr_auc", "r_precision",
            "top1_hit", "top5_overlap"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print(table[keep].round(4))
    table.to_csv(a.out)
    print("saved", a.out)


if __name__ == "__main__":
    main()
