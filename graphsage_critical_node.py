#!/usr/bin/env python3
"""
GraphSAGE criticality score model (position + kinematics + history features).

Node features are the same as for the LightGBM model (swarm_common.py). The graph is built here for every
snapshot from the agent positions only: two agents are connected if their distance is <= the sensing range
(default 55). The network output goes through a sigmoid and the loss is the MSE to the data score.
Metrics are the same as for the LightGBM model.

    python graphsage_critical_node.py --max-cases 6 --epochs 2     # quick test
    python graphsage_critical_node.py                              # mean aggregator
    python graphsage_critical_node.py --aggregators mean pool lstm
    python graphsage_critical_node.py --range 25                   # shorter sensing range
"""
import argparse
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import GraphSAGE

warnings.filterwarnings("ignore")
from swarm_common import (FEATURE_GROUPS, get_features, load_data, make_snapshots, rows_for, find_threshold,
                          all_metrics, subset_table, obstacle_cuts)


class SnapshotDS(torch.utils.data.Dataset):
    """One graph per snapshot. The edges are built from the positions when the snapshot is asked for."""
    def __init__(self, x, pos, score, offs, snap_ids, radius):
        self.x = x
        self.pos = pos
        self.score = score
        self.offs = offs
        self.ids = snap_ids
        self.radius = radius

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        s = int(self.ids[i])
        a = int(self.offs[s])
        b = int(self.offs[s + 1])
        p = self.pos[a:b]
        close = torch.cdist(p, p) <= self.radius
        close.fill_diagonal_(False)
        dst, src = close.nonzero(as_tuple=True)
        edge_index = torch.stack([src, dst])
        return Data(x=self.x[a:b], y=self.score[a:b], edge_index=edge_index, num_nodes=b - a)


def build_model(agg, in_dim, hidden, layers, dropout):
    if agg == "pool":
        kw = dict(aggr="max", project=True)
    else:
        kw = dict(aggr=agg)
    return GraphSAGE(in_channels=in_dim, hidden_channels=hidden, num_layers=layers, out_channels=1,
                     dropout=dropout, norm="BatchNorm", **kw)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        batch = batch.to(device)
        out.append(torch.sigmoid(model(batch.x, batch.edge_index).squeeze(-1)).cpu())
    return torch.cat(out).numpy()


def train_one(agg, a, loaders, in_dim, device, true_val):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    model = build_model(agg, in_dim, a.hidden, a.layers, a.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    best = 1e9
    best_state = None
    bad = 0
    t0 = time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        total = 0
        n = 0
        for batch in loaders["train"]:
            batch = batch.to(device)
            opt.zero_grad()
            pred = torch.sigmoid(model(batch.x, batch.edge_index).squeeze(-1))
            loss = torch.mean((pred - batch.y) ** 2)          # difference to the data score
            loss.backward()
            opt.step()
            total += loss.item() * batch.num_nodes
            n += batch.num_nodes
        p_val = predict(model, loaders["val"], device)
        val_mae = np.abs(p_val - true_val).mean()
        print("  [%s] epoch %02d  train mse %.4f  val mae %.4f  (%ds)" % (agg, epoch, total / n, val_mae, time.time() - t0),
              flush=True)
        if val_mae < best:
            best = val_mae
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    model.load_state_dict(best_state)
    return model, best_state, best, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=".")
    ap.add_argument("--groups", nargs="+", default=["raw", "local", "context"], choices=list(FEATURE_GROUPS))
    ap.add_argument("--no-target", action="store_true", help="drop dist_to_target and rank_dist_to_target")
    ap.add_argument("--no-history", action="store_true", help="do not add the history features")
    ap.add_argument("--range", type=float, default=55.0, help="sensing range: edge if distance <= range")
    ap.add_argument("--aggregators", nargs="+", default=["mean"])
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--train-stride", type=int, default=3, help="use every k-th snapshot for training (same as LightGBM)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-cases", type=int, default=0, help="debug: only use this many cases per split")
    ap.add_argument("--out", default="results_score_graphsage")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    out_dir = Path(a.data_dir) / a.out
    out_dir.mkdir(exist_ok=True)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    base_feats = get_features(a.groups, a.no_target)
    df, Xnp, feats = load_data(a.data_dir, base_feats, a.max_cases, history=not a.no_history)
    print(len(base_feats), "base features +", len(feats) - len(base_feats), "history features =", len(feats))
    snaps, offs = make_snapshots(df)
    cuts = obstacle_cuts(df)

    X = torch.tensor(Xnp)
    POS = torch.tensor(df[["rel_x", "rel_y", "rel_z"]].to_numpy(np.float32))      # unscaled positions, for the edges only
    SCORE = torch.tensor(df["score_true"].to_numpy(np.float32))

    sn = {}
    idx = {}
    loaders = {}
    for split, stride in [("train", a.train_stride), ("val", 1), ("test", 1)]:
        sn[split], idx[split] = rows_for(snaps, offs, split, stride)
        ds = SnapshotDS(X, POS, SCORE, offs, sn[split].ts_id.to_numpy(), a.range)
        loaders[split] = DataLoader(ds, batch_size=a.batch_size, shuffle=(split == "train"))
        print(split, ":", len(ds), "snapshots")

    val_rows = df.iloc[idx["val"]].reset_index(drop=True)
    test_rows = df.iloc[idx["test"]].reset_index(drop=True)
    true_val = val_rows["score_true"].to_numpy()

    cols = ["mae", "rmse", "r2", "spearman", "accuracy", "precision", "recall", "f1", "pr_auc", "r_precision",
            "top1_hit", "top5_overlap"]
    results = []
    best_overall = None
    for agg in a.aggregators:
        print("\n== GraphSAGE-" + agg)
        model, state, val_mae, fit_s = train_one(agg, a, loaders, len(feats), device, true_val)
        p_val = predict(model, loaders["val"], device)
        t0 = time.time()
        p_test = predict(model, loaders["test"], device)         # includes building the graphs
        ms_per_snapshot = 1000 * (time.time() - t0) / len(sn["test"])
        thr = find_threshold(val_rows["is_critical"].to_numpy(), p_val)
        m = all_metrics(test_rows, p_test, thr)
        m.update(model="GraphSAGE-" + agg, fit_s=fit_s, ms_per_snapshot=ms_per_snapshot, val_mae=val_mae)
        results.append(m)
        print("  test: MAE %.4f  R2 %.4f  PR-AUC %.4f  R-precision %.4f  (%.1f ms per snapshot)" %
              (m["mae"], m["r2"], m["pr_auc"], m["r_precision"], ms_per_snapshot))
        if best_overall is None or val_mae < best_overall["val_mae"]:
            best_overall = dict(agg=agg, state=state, val_mae=val_mae, p_val=p_val, p_test=p_test, thr=thr)

    # ---- save the best aggregator (lowest validation MAE)
    agg = best_overall["agg"]
    p_test = best_overall["p_test"]
    thr = best_overall["thr"]
    torch.save(best_overall["state"], out_dir / "graphsage_score.pt")
    res = pd.DataFrame(results).set_index("model")
    res.to_csv(out_dir / "metrics.csv")
    sub = subset_table(test_rows, p_test, thr, cuts, "GraphSAGE-" + agg)
    sub.to_csv(out_dir / "subset_metrics_test.csv", index=False)
    pred = test_rows[["case_id", "step", "agent_id", "ts_id", "is_critical", "score_true"]].copy()
    pred["score_gs"] = p_test
    pred.to_parquet(out_dir / "test_predictions.parquet", index=False)
    pred_val = val_rows[["case_id", "step", "agent_id", "ts_id", "is_critical", "score_true"]].copy()
    pred_val["score_gs"] = best_overall["p_val"]
    pred_val.to_parquet(out_dir / "val_predictions.parquet", index=False)      # used by the stacking script
    config = {"features": feats, "base_features": base_feats, "history": not a.no_history, "groups": a.groups,
              "aggregator": agg, "hidden": a.hidden, "layers": a.layers,
              "dropout": a.dropout, "range": a.range, "threshold": thr, "train_stride": a.train_stride,
              "seed": a.seed}
    json.dump(config, open(out_dir / "run_config.json", "w"), indent=2)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    print("\n=== GraphSAGE criticality score (test) ===")
    print(res[cols + ["ms_per_snapshot"]].round(4))
    print("\nbest aggregator on validation:", agg)
    print("\n=== test metrics by subset (swarm size, obstacles, attrition, agents lost) ===")
    print(sub.set_index("subset")[["n_cases", "n_rows", "mae", "r2", "accuracy", "recall", "precision", "pr_auc",
                                   "r_precision"]].round(3))
    print("\nsaved to", out_dir)


if __name__ == "__main__":
    main()
