"""
harmonize/train_head_cached.py

Train a frozen-FM classification/regression head on **precomputed** FM embeddings
(`harmonize/embed_fm.py`) instead of re-encoding the FM every epoch. The FM is
frozen, so its per-window embedding is constant → this is mathematically identical
to `train_head --backbone-fm` but ~100× faster (only the small MLP trains), and one
embedding cache serves every head × the with/without-demographics variant.

Reuses the exact label definitions (`labels_from_batch`) and metrics from
`train_head`, and writes the same config.json / train_log.jsonl / best.pt layout,
so `10_scripts/aggregate_eval.py`-style harvesting is unchanged.

Usage:
    python -m harmonize.train_head_cached --head dsofa2 \
        --emb-train data/fm_embed/chronos_bolt_base_umm_train.npy \
        --emb-val   data/fm_embed/chronos_bolt_base_umm_val.npy \
        --train-h5  data/dataset/umm_seed42_train__all_4h_fwd.h5 \
        --val-h5    data/dataset/umm_seed42_val__all_4h_fwd.h5 \
        --run-name  fmhead_chronos_bolt_base_dsofa2_umm [--head-static] [--long-parquet ...]
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from harmonize.icu_dataset import ICUDataset
from harmonize.train_head import labels_from_batch, _binary_curve, LOS_LOG_OFFSET

log = logging.getLogger("train_head_cached")
ROOT = Path(__file__).resolve().parents[1]
BINARY = ("mortality", "aki", "aki_uo", "dsofa2", "sepsis")


def _aki_ctx(ds: ICUDataset, train_h5: Path, norm_stats: Path | None) -> dict:
    dataset = Path(train_h5).name.split("_seed")[0]
    norm_path = norm_stats or (ROOT / "data" / "splits" / f"{dataset}_norm_stats.parquet")
    norm = pd.read_parquet(norm_path).set_index("variable")
    fi, ft = list(ds.features_in), list(ds.features_target)
    if "creatinine" not in fi or "creatinine" not in ft:
        raise SystemExit(f"AKI head needs creatinine in features ({dataset})")
    ctx = {"ci_in": fi.index("creatinine"), "ci_tgt": ft.index("creatinine"),
           "creat_mean": float(norm.loc["creatinine", "train_mean"]),
           "creat_std": float(norm.loc["creatinine", "train_std"]), "ui_tgt": None}
    fs = list(ds.features_static)
    if ("urine_output" in ft and "weight_admission" in fs
            and "urine_output" in norm.index and "weight_admission" in norm.index):
        ctx.update({"ui_tgt": ft.index("urine_output"), "wi": fs.index("weight_admission"),
                    "urine_mean": float(norm.loc["urine_output", "train_mean"]),
                    "urine_std": float(norm.loc["urine_output", "train_std"])})
    return ctx


def _susinf_map(long_parquet: Path | None) -> dict | None:
    if long_parquet is None or not Path(long_parquet).exists():
        raise SystemExit("--head sepsis requires --long-parquet (UMM/MIMIC only, §7.3)")
    from harmonize.scores.sepsis3 import susinf_per_stay
    ldf = pd.read_parquet(long_parquet, columns=["stay_id", "variable", "hour", "value"])
    _si = susinf_per_stay(ldf).set_index("stay_id")
    # Onset hour, not the flag — labels_from_batch enforces Sepsis-3's ±24 h coincidence
    # against it (§37.38 D19). NaN == no suspected infection.
    si = _si["susinf_onset_h"].where(_si["has_susinf"], np.nan)
    return {int(k): float(v) for k, v in si.items()}


def _collect(h5: Path, head: str, aki_ctx, susinf_map, bs: int):
    """Iterate the H5 in row order (matches embed_fm) → (label, valid, static) arrays."""
    ds = ICUDataset(str(h5), preload=True)
    ys, vs, sts = [], [], []
    for batch in DataLoader(ds, batch_size=bs, shuffle=False):
        y, valid = labels_from_batch(head, batch, aki_ctx, susinf_map)
        ys.append(y.numpy()); vs.append(valid.numpy()); sts.append(batch["static"].numpy())
    return np.concatenate(ys), np.concatenate(vs), np.concatenate(sts), ds


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train a frozen-FM head on cached embeddings")
    ap.add_argument("--head", required=True, choices=list(BINARY) + ["los"])
    ap.add_argument("--emb-train", required=True, type=Path)
    ap.add_argument("--emb-val", required=True, type=Path)
    ap.add_argument("--train-h5", required=True, type=Path)
    ap.add_argument("--val-h5", required=True, type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--output-dir", default="checkpoints")
    ap.add_argument("--head-static", action="store_true")
    ap.add_argument("--long-parquet", type=Path, default=None)
    ap.add_argument("--norm-stats", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- labels (one ordered pass over each split, same order as embed_fm) ----
    aki_ctx = susinf_map = None
    ds_probe = ICUDataset(str(args.train_h5), preload=True)
    if args.head in ("aki", "aki_uo"):
        aki_ctx = _aki_ctx(ds_probe, args.train_h5, args.norm_stats)
    if args.head == "sepsis":
        susinf_map = _susinf_map(args.long_parquet)
    y_tr, v_tr, st_tr, _ = _collect(args.train_h5, args.head, aki_ctx, susinf_map, args.batch_size)
    y_va, v_va, st_va, _ = _collect(args.val_h5, args.head, aki_ctx, susinf_map, args.batch_size)

    emb_tr = np.load(args.emb_train); emb_va = np.load(args.emb_val)
    if len(emb_tr) != len(y_tr) or len(emb_va) != len(y_va):
        raise SystemExit(f"cache/H5 row mismatch: emb {len(emb_tr)}/{len(emb_va)} "
                         f"vs labels {len(y_tr)}/{len(y_va)} — re-embed the matching H5")
    d = emb_tr.shape[1]
    n_static = st_tr.shape[1] if args.head_static else 0

    def feat(emb, st):
        return np.concatenate([emb, np.nan_to_num(st)], axis=1) if args.head_static else emb
    X_tr = torch.from_numpy(feat(emb_tr, st_tr).astype(np.float32))
    X_va = torch.from_numpy(feat(emb_va, st_va).astype(np.float32)).to(device)
    Y_tr = torch.from_numpy(y_tr.astype(np.float32))
    V_tr = torch.from_numpy(v_tr.astype(bool))
    y_va_t = torch.from_numpy(y_va.astype(np.float32)).to(device)
    v_va_np = v_va.astype(bool)

    # ---- classifier: same MLP as FrozenBackboneHead ----
    clf = nn.Sequential(nn.Linear(d + n_static, d), nn.ReLU(), nn.Dropout(args.dropout),
                        nn.Linear(d, 1)).to(device)
    binary = args.head in BINARY
    loss_fn = nn.BCEWithLogitsLoss(reduction="none") if binary else nn.MSELoss(reduction="none")
    opt = torch.optim.AdamW(clf.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    import datetime
    from harmonize.common import archive_existing_run
    started_at = datetime.datetime.now().isoformat(timespec="seconds")
    out_dir = Path(args.output_dir) / args.run_name
    archive_existing_run(out_dir)   # never silently overwrite a previous run's model
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))
    log.info("Cached head %s on %s: d=%d +static=%s | train %d (pos %.3f) val %d",
             args.head, args.run_name, d, bool(n_static), int(V_tr.sum()),
             float(Y_tr[V_tr].mean()) if binary and V_tr.any() else float("nan"), len(X_va))

    sel_dir = +1 if binary else -1               # maximise AUROC / minimise MAE
    best_score, best_epoch = -sel_dir * 1e9, -1  # init worst for the chosen direction
    n = len(X_tr); train_log = []
    for epoch in range(1, args.epochs + 1):
        _t0 = time.time()
        clf.train(); perm = torch.randperm(n); run = 0.0; ns = 0
        for s in range(0, n, args.batch_size):
            idx = perm[s:s + args.batch_size]
            xb = X_tr[idx].to(device); yb = Y_tr[idx].to(device); vb = V_tr[idx].to(device)
            if not vb.any():
                continue
            pred = clf(xb).squeeze(-1)
            loss = (loss_fn(pred, yb) * vb.float()).sum() / vb.sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); ns += 1
        # ---- val ----
        clf.eval()
        with torch.no_grad():
            pv = clf(X_va).squeeze(-1).cpu().numpy()[v_va_np]
            tv = y_va.astype(np.float32)[v_va_np]
        if binary:
            prob = 1.0 / (1.0 + np.exp(-pv))
            auroc, auprc = _binary_curve(tv, prob)
            score = auroc; row = {"epoch": epoch, "val_auroc": float(auroc),
                                  "val_auprc": float(auprc), "pos_rate": float(tv.mean())}
        else:
            mae_hours = float(np.abs(np.exp(pv) - np.exp(tv)).mean())
            score = mae_hours; row = {"epoch": epoch, "val_mae_hours": mae_hours,
                                      "val_mse_log": float(((pv - tv) ** 2).mean())}
        row["train_loss"] = run / max(ns, 1); row["elapsed_s"] = round(time.time() - _t0, 1); train_log.append(row)
        if (score - best_score) * sel_dir > 0:
            best_score, best_epoch = score, epoch
            torch.save(clf.state_dict(), out_dir / "best.pt")
    (out_dir / "train_log.jsonl").write_text("\n".join(json.dumps(r) for r in train_log))
    key = "val_auroc" if binary else "val_mae_hours"
    log.info("Done %s: best %s=%.4f @epoch %d → %s", args.run_name, key, best_score, best_epoch, out_dir)
    finished_at = datetime.datetime.now().isoformat(timespec="seconds")
    (out_dir / "final_metrics.json").write_text(json.dumps(
        {"head": args.head, "run_name": args.run_name, "head_static": bool(n_static),
         "started_at": started_at, "finished_at": finished_at,
         "best_epoch": best_epoch, key: float(best_score)}, indent=2))
    from harmonize.common import append_run_manifest
    append_run_manifest(args.run_name, out_dir, started_at, finished_at,
                        model=f"fmhead:{args.head}", dataset=Path(args.train_h5).stem.split("_seed")[0],
                        epochs_run=best_epoch, best_metric=key, best_val=float(best_score))


if __name__ == "__main__":
    main()
