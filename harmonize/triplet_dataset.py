"""
harmonize/triplet_dataset.py
PyTorch Dataset for the RAW triplet (set-function / STraTS) forecast HDF5
produced by `harmonize.stage3_make_triplets` (schema 1.0-forecast-triplet).

Input is a variable-length SET of observations per window, stored CSR/ragged:
    tri_time   (Σ tokens,)  float32   Δt = hour − t0 ∈ [0, window_in)
    tri_var    (Σ tokens,)  int16     variable index into features_in
    tri_val    (Σ tokens,)  float32   z-scored value (same μ/σ as dense)
    tri_offsets(N+1,)       int64     per-window CSR offsets
The target and all metadata are identical to the dense pipeline (copied verbatim
by the builder), so loss/evaluation are directly comparable.

`triplet_collate` pads each batch to its own max token count and builds a
key-padding mask (True = padded), so no global cap or wasted padding is stored.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


def _decode(arr) -> list[str]:
    return [s.decode() if isinstance(s, bytes) else str(s) for s in arr]


class TripletICUDataset(Dataset):
    def __init__(self, h5_path: str | Path, preload: bool = True) -> None:
        self.h5_path = Path(h5_path)
        with h5py.File(self.h5_path, "r") as f:
            a = f.attrs
            self.features_in     = _decode(a["features_in"])
            self.features_target = _decode(a["features_target"])
            self.features_static = _decode(a["features_static"])
            self.window_in   = int(a["window_in"])
            self.window_out  = int(a["window_out"])
            self.dataset_name = str(a["dataset"])
            self.split        = str(a["split"])
            self.seed         = int(a["seed"])
            self._offsets = f["tri_offsets"][:]
            self._n = self._offsets.shape[0] - 1
            if preload:
                self._tri_time = f["tri_time"][:]
                self._tri_var  = f["tri_var"][:].astype(np.int64)
                self._tri_val  = f["tri_val"][:]
                self._x_target    = f["x_target"][:]
                self._mask_target = f["mask_target"][:]
                self._static      = f["static"][:]
                self._static_mask = f["static_mask"][:]
                self._stay_id   = f["stay_id"][:]
                self._ds_id     = f["ds_id"][:]
                self._t0_hour   = f["t0_hour"][:]
                self._los_hours = f["los_hours"][:]
                self._mort_hours = (f["mortality_hours"][:] if "mortality_hours" in f
                                    else np.full(self._n, np.nan, np.float32))
        self._preloaded = preload
        if not preload:
            self._h5 = None

    # --- size / features ---
    def __len__(self) -> int:
        return self._n

    @property
    def n_features_in(self) -> int:
        return len(self.features_in)

    @property
    def n_features_target(self) -> int:
        return len(self.features_target)

    @property
    def n_features_static(self) -> int:
        return len(self.features_static)

    def target_indices_in_input(self) -> list[int]:
        return [self.features_in.index(f) for f in self.features_target]

    # --- item ---
    def __getitem__(self, k: int) -> dict:
        lo = int(self._offsets[k]); hi = int(self._offsets[k + 1])
        if self._preloaded:
            t = self._tri_time[lo:hi]; v = self._tri_var[lo:hi]; val = self._tri_val[lo:hi]
            x_target = self._x_target[k]; mask_target = self._mask_target[k]
            static = self._static[k]; static_mask = self._static_mask[k]
            stay_id = int(self._stay_id[k]); ds_id = int(self._ds_id[k])
            t0_hour = float(self._t0_hour[k]); los = float(self._los_hours[k])
            mort = float(self._mort_hours[k])
        else:
            if self._h5 is None:
                self._h5 = h5py.File(self.h5_path, "r")
            f = self._h5
            t = f["tri_time"][lo:hi]; v = f["tri_var"][lo:hi].astype(np.int64); val = f["tri_val"][lo:hi]
            x_target = f["x_target"][k]; mask_target = f["mask_target"][k]
            static = f["static"][k]; static_mask = f["static_mask"][k]
            stay_id = int(f["stay_id"][k]); ds_id = int(f["ds_id"][k])
            t0_hour = float(f["t0_hour"][k]); los = float(f["los_hours"][k])
            mort = float(f["mortality_hours"][k]) if "mortality_hours" in f else float("nan")
        return {
            "tri_time": torch.from_numpy(t.astype(np.float32)),
            "tri_var":  torch.from_numpy(v.astype(np.int64)),
            "tri_val":  torch.from_numpy(val.astype(np.float32)),
            "x_target":    torch.from_numpy(x_target.astype(np.float32)),
            "mask_target": torch.from_numpy(np.asarray(mask_target)),
            "static":      torch.from_numpy(static.astype(np.float32)),
            "static_mask": torch.from_numpy(np.asarray(static_mask)),
            "stay_id":  torch.tensor(stay_id, dtype=torch.int64),
            "ds_id":    torch.tensor(ds_id, dtype=torch.int64),
            "t0_hour":  torch.tensor(t0_hour, dtype=torch.float32),
            "los_hours": torch.tensor(los, dtype=torch.float32),
            "mortality_hours": torch.tensor(mort, dtype=torch.float32),
        }


def triplet_collate(batch: list[dict]) -> dict:
    """Pad the ragged triplet sets to the batch max; build tri_pad (True=pad)."""
    B = len(batch)
    lens = [b["tri_time"].shape[0] for b in batch]
    M = max(1, max(lens))
    tri_time = torch.zeros(B, M, dtype=torch.float32)
    tri_var  = torch.zeros(B, M, dtype=torch.int64)
    tri_val  = torch.zeros(B, M, dtype=torch.float32)
    tri_pad  = torch.ones(B, M, dtype=torch.bool)   # True = padded
    for i, b in enumerate(batch):
        n = lens[i]
        if n:
            tri_time[i, :n] = b["tri_time"]
            tri_var[i, :n]  = b["tri_var"]
            tri_val[i, :n]  = b["tri_val"]
            tri_pad[i, :n]  = False
    out = {
        "tri_time": tri_time, "tri_var": tri_var, "tri_val": tri_val, "tri_pad": tri_pad,
        "x_target":    torch.stack([b["x_target"] for b in batch]),
        "mask_target": torch.stack([b["mask_target"] for b in batch]),
        "static":      torch.stack([b["static"] for b in batch]),
        "static_mask": torch.stack([b["static_mask"] for b in batch]),
        "stay_id":  torch.stack([b["stay_id"] for b in batch]),
        "ds_id":    torch.stack([b["ds_id"] for b in batch]),
        "t0_hour":  torch.stack([b["t0_hour"] for b in batch]),
        "los_hours": torch.stack([b["los_hours"] for b in batch]),
        "mortality_hours": torch.stack([b["mortality_hours"] for b in batch]),
    }
    return out
