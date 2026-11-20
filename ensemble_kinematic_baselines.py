#!/usr/bin/env python3
"""
LightGBM criticality score model (position + kinematics + history features, no graph).

Score = p * s1 + (1 - p) * s0, where p is the probability of being critical and s1 / s0 are LightGBM
regressors of the data score for critical / decoy agents (TwoPartModel in swarm_common.py).
The threshold for critical / not critical is chosen on validation. Also writes the metrics, the key
factors (feature importance) and the timing.

    python ensemble_kinematic_baselines.py --max-cases 6      # quick test
    python ensemble_kinematic_baselines.py
    python ensemble_kinematic_baselines.py --no-history       # single snapshot features only
"""
import argparse
import json
import time
import warnings
from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
from swarm_common import (FEATURE_GROUPS, get_features, load_data, make_snapshots, rows_for, find_threshold,
                          all_metrics, subset_table, obstacle_cuts, TwoPartModel)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--groups", nargs="+", default=["raw", "local", "context"], choices=list(FEATURE_GROUPS))
    ap.add_argument("--no-target", action="store_true", help="drop dist_to_target and rank_dist_to_target")
    ap.add_argument("--no-history", action="store_true", help="do not add the history features")
    ap.add_argument("--train-stride", type=int, default=3, help="use every k-th snapshot for training (same as GraphSAGE)")
    ap.add_argument("--n-estimators", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=0, help="debug: only use this many cases per split")
    ap.add_argument("--out", default="results_score_lightgbm")
    a = ap.parse_args()

    out_dir = Path(a.data_dir) / a.out
    out_dir.mkdir(exist_ok=True)

    base_feats = get_features(a.groups, a.no_target)
    df, X, feats = load_data(a.data_dir, base_feats, a.max_cases, history=not a.no_history)
    print(len(base_feats), "base features +", len(feats) - len(base_feats), "history features =", len(feats))
    snaps, offs = make_snapshots(df)
    cuts = obstacle_cuts(df)

    s_train, i_train = rows_for(snaps, offs, "train", a.train_stride)
    s_val, i_val = rows_for(snaps, offs, "val", 1)
    s_test, i_test = rows_for(snaps, offs, "test", 1)
    print("snapshots: train", len(s_train), "val", len(s_val), "test", len(s_test))

    # ---- train: score = p * s1 + (1 - p) * s0
    y = df["is_critical"].to_numpy()
    score = df["score_true"].to_numpy()
    model = TwoPartModel()
    t0 = time.time()
    iters = model.fit(X[i_train], y[i_train], score[i_train], X[i_val], y[i_val], score[i_val],
                      seed=a.seed, n_estimators=a.n_estimators, patience=a.patience)
    fit_s = time.time() - t0
    print("trained in", round(fit_s), "s, best iterations", iters)

    # ---- predict
    p_val = model.predict(X[i_val])
    t0 = time.time()
    p_test = model.predict(X[i_test])
    infer_s = time.time() - t0
    ms_per_snapshot = 1000 * infer_s / len(s_test)
    print("scoring time:", round(ms_per_snapshot, 2), "ms per snapshot (features already computed)")

    val_rows = df.iloc[i_val].reset_index(drop=True)
    test_rows = df.iloc[i_test].reset_index(drop=True)
    thr = find_threshold(val_rows["is_critical"].to_numpy(), p_val)      # threshold from validation only

    # ---- metrics
    res_val = all_metrics(val_rows, p_val, thr)
    res_test = all_metrics(test_rows, p_test, thr)
    res_test.update(fit_s=fit_s, ms_per_snapshot=ms_per_snapshot)
    metrics = pd.DataFrame([dict(res_val, split="val"), dict(res_test, split="test")]).set_index("split")
    metrics.to_csv(out_dir / "metrics.csv")
    sub = subset_table(test_rows, p_test, thr, cuts, "LightGBM")
    sub.to_csv(out_dir / "subset_metrics_test.csv", index=False)

    cols = ["mae", "rmse", "r2", "spearman", "accuracy", "precision", "recall", "f1", "pr_auc", "r_precision",
            "top1_hit", "top5_overlap"]
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print("\n=== LightGBM criticality score ===")
    print(metrics[cols].round(4))
    print("\n=== test metrics by subset (swarm size, obstacles, attrition, agents lost) ===")
    print(sub.set_index("subset")[["n_cases", "n_rows", "mae", "r2", "accuracy", "recall", "precision", "pr_auc",
                                   "r_precision"]].round(3))

    # ---- key factors: feature importance and correlation with the data score
    gain = pd.Series(model.gain_importance(), index=feats)
    sample = i_train[::5]
    corr = []
    for j in range(len(feats)):
        corr.append(pd.Series(X[sample, j]).corr(pd.Series(score[sample]), method="spearman"))
    factors = pd.DataFrame({"gain_share": gain, "spearman_with_score": corr}, index=feats)
    factors = factors.sort_values("gain_share", ascending=False)
    factors.to_csv(out_dir / "key_factors.csv")
    print("\n=== key factors (top 12 by gain) ===")
    print(factors.head(12).round(4))

    # ---- save the model and the predictions
    model.save(out_dir)
    pred = test_rows[["case_id", "step", "agent_id", "ts_id", "is_critical", "score_true"]].copy()
    pred["score_lgb"] = p_test
    pred.to_parquet(out_dir / "test_predictions.parquet", index=False)
    pred_val = val_rows[["case_id", "step", "agent_id", "ts_id", "is_critical", "score_true"]].copy()
    pred_val["score_lgb"] = p_val
    pred_val.to_parquet(out_dir / "val_predictions.parquet", index=False)      # used by the stacking script
    config = {"features": feats, "base_features": base_feats, "history": not a.no_history, "groups": a.groups,
              "threshold": thr, "train_stride": a.train_stride, "iterations": iters, "seed": a.seed}
    json.dump(config, open(out_dir / "run_config.json", "w"), indent=2)
    print("\nsaved to", out_dir)


if __name__ == "__main__":
    main()
