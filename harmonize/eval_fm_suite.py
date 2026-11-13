"""
harmonize/eval_fm_suite.py
Stage-5 evaluation of a *suite* of zero-shot foundation-model baselines on ONE
dataset, loading the test set / norm-stats / susinf table exactly **once** and
looping over the models in a single process (we only need inference, so there is
no reason to pay the ~2 GB preload per model).

Each model writes its own `docs/tables/eval_<run>.json` (same schema as
eval_forecast), so `10_scripts/aggregate_eval.py` picks them up unchanged.

Usage:
    python -m harmonize.eval_fm_suite \
        --test-h5 data/dataset/<ds>_seed42_test__all_4h_fwd.h5 \
        [--long-parquet data/long/<ds>_long.parquet] \
        [--models "chronos:amazon/chronos-bolt-tiny, chronos:amazon/chronos-bolt-mini,
                   chronos:amazon/chronos-bolt-small, chronos:amazon/chronos-bolt-base,
                   chronos2:amazon/chronos-2"] \
        [--point median] [--batch-size 256] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from harmonize.icu_dataset import ICUDataset
from harmonize.eval_forecast import build_fm, evaluate, load_susinf, ROOT

log = logging.getLogger("eval_fm_suite")

# Default suite: the full Chronos-Bolt size axis + the multivariate Chronos-2.
DEFAULT_MODELS = [
    ("chronos",  "amazon/chronos-bolt-tiny"),
    ("chronos",  "amazon/chronos-bolt-mini"),
    ("chronos",  "amazon/chronos-bolt-small"),
    ("chronos",  "amazon/chronos-bolt-base"),
    ("chronos2", "amazon/chronos-2"),
]


def _parse_models(spec: str | None) -> list[tuple[str, str]]:
    if not spec:
        return DEFAULT_MODELS
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        fm, _, mid = item.partition(":")
        out.append((fm.strip(), mid.strip()))
    return out


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Evaluate a suite of FM baselines on one dataset (single data load)")
    ap.add_argument("--test-h5", required=True, type=Path)
    ap.add_argument("--norm-stats", type=Path, default=None)
    ap.add_argument("--long-parquet", type=Path, default=None)
    ap.add_argument("--models", type=str, default=None,
                    help="comma-separated 'fm:hf_id' list; default = full bolt axis + chronos-2")
    ap.add_argument("--point", type=str, default="median", choices=["median", "mean"])
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs" / "tables")
    ap.add_argument("--dump-preds", type=Path, default=None, metavar="DIR",
                    help="write each FM's forecast trajectory as .npy instead of evaluating it. "
                         "For the outcome-head trajectory arms (§16.10a): the FMs are zero-shot, "
                         "so unlike denseAR/LightGBM/the ensemble their train-split predictions "
                         "are NOT in-sample, which is what makes them the clean rungs of that "
                         "ladder.")
    ap.add_argument("--rows", choices=["all", "first-day"], default="all",
                    help="restrict to admission windows (t0_hour == 0), ascending dataset order, "
                         "and tag the dump '_fd'. Matches ensemble_eval --rows and "
                         "train_head --first-day-only.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.test_h5.stem.split("_seed")[0]
    norm_path = args.norm_stats or (ROOT / "data" / "splits" / f"{dataset}_norm_stats.parquet")

    # ---- load everything ONCE ----
    ds = ICUDataset(str(args.test_h5), preload=True)
    norm_stats = pd.read_parquet(norm_path).set_index("variable")
    susinf = load_susinf(args.long_parquet)
    models = _parse_models(args.models)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Suite on %s (%d windows, %s) — %d models, sepsis3=%s",
             dataset, len(ds), device, len(models), susinf is not None)

    split = args.test_h5.stem.split("__")[0].split("_")[-1]
    sel = None
    if args.rows == "first-day":
        import h5py
        with h5py.File(args.test_h5, "r") as f:
            sel = np.flatnonzero(f["t0_hour"][:] == 0)
        if sel.size == 0:
            raise SystemExit("--rows first-day selected 0 windows; is t0_hour present?")
        log.info("--rows first-day: %d -> %d windows", len(ds), sel.size)

    for fm, mid in models:
        model, cfg = build_fm(fm, ds, device, mid, args.point)
        log.info("→ %s (%s)", cfg["run_name"], mid)
        if args.dump_preds is not None:
            args.dump_preds.mkdir(parents=True, exist_ok=True)
            idx_all = np.arange(len(ds)) if sel is None else sel
            tag = "_fd" if sel is not None else ""
            dest = args.dump_preds / f"{cfg['run_name']}__{dataset}_{split}{tag}.npy"
            if dest.exists():
                log.info("   skip %s (exists)", dest.name)
            else:
                buf = np.empty((len(idx_all), ds.window_out, ds.n_features_target),
                               dtype=np.float32)
                for s in range(0, len(idx_all), args.batch_size):
                    i = idx_all[s:s + args.batch_size]
                    with torch.no_grad():
                        out = model(torch.from_numpy(ds._x_in[i]).to(device),
                                    torch.from_numpy(ds._mask_in[i]).to(device),
                                    torch.from_numpy(ds._static[i]).to(device))
                    buf[s:s + len(i)] = out.cpu().numpy().astype(np.float32)
                np.save(dest, buf)
                log.info("   wrote %s  %s", dest.name, buf.shape)
            del model
            continue
        summary = evaluate(model, cfg, ds, norm_stats, dataset, device,
                           susinf=susinf, batch_size=args.batch_size, limit=args.limit)
        summary["test_h5"] = str(args.test_h5)
        out = args.out_dir / f"eval_{cfg['run_name']}_{dataset}.json"
        out.write_text(json.dumps(summary, indent=2))
        log.info("   wrote %s", out)
        # free GPU between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
