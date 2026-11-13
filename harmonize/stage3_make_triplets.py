"""
harmonize/stage3_make_triplets.py
Stage 3 (triplet variant) — RAW set-function input for the true STraTS model.

The dense Stage-3 (`stage3_make_dataset.py`) aggregates each input window to an
hourly grid. A set-function / triplet model (STraTS) must instead see the *raw*
irregular observations — aggregating before the model defeats its purpose. This
module builds, for the SAME windows as an existing dense HDF5, a raw-triplet input
while copying the target and all metadata verbatim (so windows, targets, loss and
evaluation are byte-identical and fully comparable across models — only the input
representation changes).

Per window the input becomes the variable-length set of observations with
`hour ∈ [t0, t0 + window_in)`, each a triplet:
    (Δt = hour − t0 ∈ [0, window_in),  variable index,  z-scored value)
Values are z-scored with the SAME train mean/std (`{ds}_norm_stats.parquet`) the
dense pipeline uses, so the input scale matches.

Redundancy compression (set-length cap, NOT aggregation):
  Raw monitoring repeats unchanged values; carrying every repeat is wasteful and
  blows up the set length. We cap each window at the per-dataset p99 of the raw
  token count (computed over train+val+test). Windows above the cap are reduced
  by deleting *redundant* observations, escalating only as far as needed:
    Stage 1 — within a variable's time-ordered values, drop any observation equal
              to BOTH neighbours (3-in-a-row equal → middle gone; long flat runs →
              only the two endpoints survive).
    Stage 2 — if still over cap, collapse equal consecutive pairs (drop the older).
    Fallback — if still over cap, drop the oldest observations.
  Windows at/below the cap keep every raw observation. The full (uncapped) set is
  the natural object; we store the compressed form so training/storage are bounded.

Storage is CSR/ragged (no per-window padding): flat `tri_time/tri_var/tri_val`
plus `tri_offsets` (N+1). The Dataset slices per window; a collate pads per batch.

Output: data/dataset/{ds}_seed{seed}_{split}__{tag_raw}.h5   (tag_raw = dense tag
with the imputation token replaced by `raw`, e.g. all_4h_fwd → all_4h_raw).

Usage:
    python -m harmonize.stage3_make_triplets --dense-tag all_4h_fwd --datasets umm --seed 42
"""

from __future__ import annotations

import argparse
import gc
import logging
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

log = logging.getLogger(__name__)

ROOT       = Path(__file__).resolve().parents[1]
SPLITS_DIR = ROOT / "data" / "splits"
LONG_DIR   = ROOT / "data" / "long"
OUTPUT_DIR = ROOT / "data" / "dataset"
SCHEMA_VERSION = "1.0-forecast-triplet"

DATASETS = ["mimic_iii", "mimic_iv", "hirid", "eicu", "umm"]
CAP_PERCENTILE = 99.0

# Copied verbatim from the dense HDF5 (everything except the dense x_in/mask_in).
COPY_DATASETS = [
    "x_target", "mask_target", "static", "static_mask",
    "stay_id", "ds_id", "t0_hour", "los_hours", "mortality_hours",
    "sofa_scores", "target_obs_count",
]
COPY_ATTRS = [
    "window_in", "window_out", "stride", "features_in", "features_target",
    "features_static", "dataset", "split", "seed", "sofa_score_names",
]


def _long_path(ds: str) -> Path:
    for name in (f"{ds}_long_complete.parquet", f"{ds}_long.parquet"):
        p = LONG_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"no long parquet for {ds} in {LONG_DIR}")


def _dense_h5(ds: str, split: str, seed: int, dense_tag: str) -> Path:
    return OUTPUT_DIR / f"{ds}_seed{seed}_{split}__{dense_tag}.h5"


def _load_by_stay(ds: str, needed_stays, feat_idx, tmean, tstd):
    """Read raw long obs for the given stays + input features, z-scored; return
    {stay_id: (hours, var_idx, z_value)} each sorted by ascending hour."""
    lp = _long_path(ds)
    features_in = list(feat_idx)
    tbl = pq.read_table(lp, columns=["stay_id", "hour", "variable", "value"],
                        filters=[("variable", "in", features_in)])
    raw = tbl.to_pandas(); del tbl
    raw = raw[raw["stay_id"].isin(needed_stays)]
    raw = raw[raw["value"].notna() & raw["hour"].notna()]
    raw["vi"] = raw["variable"].map(feat_idx)
    raw = raw[raw["vi"].notna()]
    raw["vi"] = raw["vi"].astype(np.int32)
    mu = raw["variable"].map(tmean).to_numpy(np.float32)
    sd = raw["variable"].map(tstd).to_numpy(np.float32)
    raw["vz"] = ((raw["value"].to_numpy(np.float32) - mu) / sd).astype(np.float32)
    raw = raw[["stay_id", "hour", "vi", "vz"]].sort_values(["stay_id", "hour"])
    by_stay = {sid: (g["hour"].to_numpy(np.float32),
                     g["vi"].to_numpy(np.int32),
                     g["vz"].to_numpy(np.float32))
               for sid, g in raw.groupby("stay_id", sort=False)}
    del raw
    return by_stay


def _build_flat(stay_id, t0_hour, window_in, by_stay):
    """Uncapped CSR triplets in dense-window order. Returns (time,var,val,offsets,counts)."""
    N = stay_id.shape[0]
    tchunks, vchunks, zchunks = [], [], []
    offsets = np.zeros(N + 1, dtype=np.int64)
    counts = np.zeros(N, dtype=np.int32)
    for i in range(N):
        ent = by_stay.get(stay_id[i])
        if ent is not None:
            hours, vis, vzs = ent
            t0 = float(t0_hour[i])
            lo = np.searchsorted(hours, t0, side="left")
            hi = np.searchsorted(hours, t0 + window_in, side="left")
            if hi > lo:
                tchunks.append((hours[lo:hi] - t0).astype(np.float32))
                vchunks.append(vis[lo:hi]); zchunks.append(vzs[lo:hi])
                counts[i] = hi - lo
        offsets[i + 1] = offsets[i] + counts[i]
    tri_time = np.concatenate(tchunks) if tchunks else np.zeros(0, np.float32)
    tri_var  = np.concatenate(vchunks) if vchunks else np.zeros(0, np.int32)
    tri_val  = np.concatenate(zchunks) if zchunks else np.zeros(0, np.float32)
    return tri_time, tri_var, tri_val, offsets, counts


def _compress_window(t, v, val, cap):
    """Reduce one window's triplets (time-sorted) to <= cap by deleting redundant
    observations; see module docstring. Returns (t, v, val)."""
    n = len(t)
    if n <= cap:
        return t, v, val
    order = np.argsort(v, kind="stable")   # group by var, time order kept within
    sv = v[order]
    rm = np.zeros(n, dtype=bool)
    # group boundaries
    groups = []
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sv[j + 1] == sv[i]:
            j += 1
        groups.append(order[i:j + 1])      # positions of this var, in time order
        i = j + 1
    # Stage 1 — drop observations equal to both same-var neighbours
    for gp in groups:
        if len(gp) >= 3:
            gv = val[gp]
            mid = (gv[:-2] == gv[1:-1]) & (gv[1:-1] == gv[2:])   # interior of const run
            rm[gp[1:-1][mid]] = True
    if n - rm.sum() <= cap:
        keep = ~rm
        return t[keep], v[keep], val[keep]
    # Stage 2 — collapse equal consecutive pairs (drop the older)
    for gp in groups:
        kept = gp[~rm[gp]]                  # still-kept positions, time order
        if len(kept) >= 2:
            gv = val[kept]
            dup = gv[1:] == gv[:-1]         # cur == prev
            rm[kept[:-1][dup]] = True       # remove the older of each equal pair
    if n - rm.sum() <= cap:
        keep = ~rm
        return t[keep], v[keep], val[keep]
    # Fallback — drop oldest still-kept tokens
    keep = ~rm
    kept_idx = np.where(keep)[0]            # time-sorted (arrays are time-sorted)
    excess = len(kept_idx) - cap
    if excess > 0:
        keep[kept_idx[:excess]] = False
    return t[keep], v[keep], val[keep]


def _compress_flat(tri_time, tri_var, tri_val, offsets, counts, cap):
    """Apply per-window compression to a CSR triplet set; rebuild CSR. Returns
    (time, var, val, offsets, new_counts, n_compressed)."""
    N = counts.shape[0]
    over = np.where(counts > cap)[0]
    if len(over) == 0:
        return tri_time, tri_var, tri_val, offsets, counts, 0
    new_counts = counts.copy()
    # only the over-cap windows change; assemble new flat by slicing per window
    tchunks, vchunks, zchunks = [], [], []
    new_off = np.zeros(N + 1, dtype=np.int64)
    over_set = set(over.tolist())
    for i in range(N):
        a, b = int(offsets[i]), int(offsets[i + 1])
        t, v, val = tri_time[a:b], tri_var[a:b], tri_val[a:b]
        if i in over_set:
            t, v, val = _compress_window(t, v, val, cap)
            new_counts[i] = len(t)
        if len(t):
            tchunks.append(t); vchunks.append(v); zchunks.append(val)
        new_off[i + 1] = new_off[i] + new_counts[i]
    return (np.concatenate(tchunks) if tchunks else np.zeros(0, np.float32),
            np.concatenate(vchunks) if vchunks else np.zeros(0, np.int32),
            np.concatenate(zchunks) if zchunks else np.zeros(0, np.float32),
            new_off, new_counts, len(over))


def _write_split(ds, split, seed, dense_tag, raw_tag,
                 tri_time, tri_var, tri_val, offsets, cap):
    dense = _dense_h5(ds, split, seed, dense_tag)
    out_path = OUTPUT_DIR / f"{ds}_seed{seed}_{split}__{raw_tag}.h5"
    N = offsets.shape[0] - 1
    with h5py.File(dense, "r") as src, h5py.File(out_path, "w") as dst:
        for name in COPY_DATASETS:
            if name in src:
                dst.create_dataset(name, data=src[name][:],
                                   compression="gzip", compression_opts=4, chunks=True)
        dst.create_dataset("tri_time", data=tri_time, compression="gzip",
                           compression_opts=4, chunks=True)
        dst.create_dataset("tri_var", data=tri_var.astype(np.int16), compression="gzip",
                           compression_opts=4, chunks=True)
        dst.create_dataset("tri_val", data=tri_val, compression="gzip",
                           compression_opts=4, chunks=True)
        dst.create_dataset("tri_offsets", data=offsets, compression="gzip", compression_opts=4)
        for a in COPY_ATTRS:
            if a in src.attrs:
                dst.attrs[a] = src.attrs[a]
        dst.attrs["imputation"]     = "raw"
        dst.attrs["representation"] = "triplet"
        dst.attrs["version"]        = SCHEMA_VERSION
        dst.attrs["n_windows"]      = N
        dst.attrs["max_tokens"]     = int(cap)
    log.info("[%s/%s]   wrote %s", ds, split, out_path.name)


def process_dataset(ds: str, dense_tag: str, seed: int, cap_override: int = 0) -> int:
    raw_tag = "_".join(dense_tag.split("_")[:-1] + ["raw"])
    splits = [s for s in ("train", "val", "test")
              if _dense_h5(ds, s, seed, dense_tag).exists()]
    if not splits:
        log.warning("[%s] no dense h5 for tag %s — skip", ds, dense_tag)
        return 0

    # norm stats + feature map from the first split's dense h5
    with h5py.File(_dense_h5(ds, splits[0], seed, dense_tag), "r") as h5:
        features_in = [s.decode() if isinstance(s, bytes) else str(s)
                       for s in h5.attrs["features_in"]]
        window_in = int(h5.attrs["window_in"])
    feat_idx = {f: i for i, f in enumerate(features_in)}
    norm = pq.read_table(SPLITS_DIR / f"{ds}_norm_stats.parquet").to_pandas().set_index("variable")
    tmean = {f: float(norm.loc[f, "train_mean"]) if f in norm.index else 0.0 for f in features_in}
    tstd  = {f: float(norm.loc[f, "train_std"])  if f in norm.index else 1.0 for f in features_in}

    def _build_split(split):
        with h5py.File(_dense_h5(ds, split, seed, dense_tag), "r") as h5:
            stay_id = h5["stay_id"][:]; t0_hour = h5["t0_hour"][:]
        t0 = time.time()
        by_stay = _load_by_stay(ds, pd.unique(stay_id), feat_idx, tmean, tstd)
        flat = _build_flat(stay_id, t0_hour, window_in, by_stay)
        del by_stay; gc.collect()
        c = flat[4]
        log.info("[%s/%s] built %d windows, raw tokens mean=%.0f p99=%.0f max=%d (%.1fs)",
                 ds, split, len(c), float(c.mean()) if len(c) else 0,
                 float(np.percentile(c, 99)) if len(c) else 0, int(c.max()) if len(c) else 0,
                 time.time() - t0)
        return flat

    def _compress_write(split, flat, cap):
        tri_time, tri_var, tri_val, offsets, counts = flat
        tri_time, tri_var, tri_val, offsets, ncounts, n_comp = _compress_flat(
            tri_time, tri_var, tri_val, offsets, counts, cap)
        log.info("[%s/%s] compressed %d/%d windows over cap; tokens %d→%d (max now %d)",
                 ds, split, n_comp, len(counts), int(counts.sum()), int(ncounts.sum()),
                 int(ncounts.max()) if len(ncounts) else 0)
        _write_split(ds, split, seed, dense_tag, raw_tag,
                     tri_time, tri_var, tri_val, offsets, cap)
        return offsets.shape[0] - 1

    # With an explicit cap, stream per split (low memory). Otherwise hold all
    # splits to compute the per-dataset p99 cap, then compress + write.
    total = 0
    if cap_override > 0:
        log.info("[%s] using cap override = %d tokens", ds, cap_override)
        for split in splits:
            total += _compress_write(split, _build_split(split), cap_override)
            gc.collect()
        return total

    built = {split: _build_split(split) for split in splits}
    cap = int(np.ceil(np.percentile(np.concatenate([built[s][4] for s in splits]),
                                     CAP_PERCENTILE)))
    log.info("[%s] per-dataset cap = p%.0f = %d tokens", ds, CAP_PERCENTILE, cap)
    for split in splits:
        total += _compress_write(split, built[split], cap)
        built[split] = None; gc.collect()
    return total


def run(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    total = 0
    for ds in args.datasets:
        total += process_dataset(ds, args.dense_tag, args.seed, cap_override=args.cap)
    log.info("Stage-3 triplet done. Total windows: %d", total)


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3 — raw triplet input from dense windows")
    p.add_argument("--dense-tag", default="all_4h_fwd")
    p.add_argument("--datasets", nargs="+", default=DATASETS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cap", type=int, default=0,
                   help="Override the per-dataset p99 token cap with a fixed value "
                        "(e.g. 2048 for HiRID, whose 2-min p99 is too large for the "
                        "AR decoder). 0 = compute p99 per dataset.")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
