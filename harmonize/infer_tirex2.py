"""
harmonize/infer_tirex2.py  —  RUN WITH THE py313_tirex ENV, NOT py313_tsf.

    python -m harmonize.infer_tirex2 ...

TiRex-2 (Podest et al. 2026, arXiv:2607.01204) is the newest **natively multivariate**
zero-shot forecasting FM: an xLSTM backbone with a variate mixer + asymmetric group
attention, 38.4 M active parameters univariate / +44.1 M multivariate. It pins
`xlstm~=2.0.3`, which drags torch back to 2.9.1 and numpy to 2.1.x — incompatible with
py313_tsf (torch 2.10 / numpy 2.3). So it takes the **same route as Moirai**: a
standalone script (h5py + numpy + torch + tirex2 only — it does NOT import the
harmonize package) that writes z-scored predictions to a .npy in H5 row order, which
the py313 eval consumes via `eval_forecast --pred-file`. Identical metrics, so TiRex-2
sits in the same table as Chronos-2 and Moirai (all three native multivariate).

Conventions (match models/chronos2.py and infer_moirai.py):
- z-scored in / z-scored out. TiRex-2 instance-norms per variate over the observed
  context, so it returns the forecast in the input's z-score space.
- Missingness is passed as **NaN** at mask=False (TiRex-2 handles NaN gaps natively via
  its nan-aware scaler), so the 0-filled H5 values never masquerade as real z=0
  observations. A channel with NO observation in the window would leave the scaler
  undefined — those fall back to the z-score mean (0), exactly as chronos2.py does.
- point forecast = the 0.5 quantile (TiRex-2 emits 9 quantiles per step).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
import torch


def _decode(arr):
    return [x.decode() if isinstance(x, (bytes, bytearray)) else str(x) for x in arr]


def _median_index(model, n_q: int) -> int:
    """Index of the 0.5 quantile among the model's outputs (fall back to the middle)."""
    for attr in ("quantile_levels", "quantiles", "output_quantiles"):
        q = getattr(model, attr, None)
        if q is not None:
            q = list(q)
            if len(q) == n_q:
                return min(range(n_q), key=lambda i: abs(float(q[i]) - 0.5))
    return n_q // 2


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot TiRex-2 forecasts → .npy (run in py313_tirex)")
    ap.add_argument("--test-h5", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output .npy (N, T_out, F_target), z-scored")
    ap.add_argument("--model", default="NX-AI/TiRex-2", help="HF id of the TiRex-2 checkpoint")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    with h5py.File(args.test_h5, "r") as f:
        feats_in = _decode(f.attrs["features_in"])
        feats_tgt = _decode(f.attrs["features_target"])
        T_in = int(f.attrs["window_in"]); T_out = int(f.attrs["window_out"])
        N = f["x_in"].shape[0]
        if args.limit:
            N = min(N, args.limit)
        x_in = f["x_in"][:N]                      # (N,T_in,F_in) z-scored, 0-filled
        mask_in = f["mask_in"][:N].astype(bool)
        ds_name = str(f.attrs["dataset"])

    tgt_idx = [feats_in.index(c) for c in feats_tgt]   # target ⊆ input
    F = len(tgt_idx)
    ctx_all = x_in[:, :, tgt_idx].astype(np.float32)   # (N,T_in,F)
    obs_all = mask_in[:, :, tgt_idx]                   # (N,T_in,F) bool
    # NaN at unobserved steps so the scaler sees true gaps, not fabricated z=0 values
    ctx_all = np.where(obs_all, ctx_all, np.nan)
    # a channel with zero observations has no scale — hand TiRex-2 the z-score mean
    never_obs = ~obs_all.any(axis=1)                   # (N,F)
    ctx_all[np.broadcast_to(never_obs[:, None, :], ctx_all.shape)] = 0.0

    from tirex2 import TimeseriesType, load_model
    model = load_model(args.model, device=device)
    print(f"TiRex-2 {args.model} on {ds_name}: {N} windows, F={F}, "
          f"ctx={T_in}->{T_out}, device={device}", flush=True)

    preds = np.zeros((N, T_out, F), dtype=np.float32)
    q_idx = None
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            # TiRex-2 wants (n_targets, context_length) per series → transpose our (T_in, F)
            batch = [TimeseriesType(target=torch.from_numpy(ctx_all[i].T).float(),
                                    past_covariates=None, future_covariates=None)
                     for i in range(s, e)]
            fc = model.forecast(timeseries=batch, prediction_length=T_out, output_type="numpy")
            for j, arr in enumerate(fc):
                arr = np.asarray(arr)                  # (F, n_quantiles, T_out)
                if q_idx is None:
                    q_idx = _median_index(model, arr.shape[1])
                    print(f"  median quantile index {q_idx} of {arr.shape[1]}", flush=True)
                preds[s + j] = arr[:, q_idx, :].T      # → (T_out, F)
            if s % (args.batch_size * 20) == 0:
                print(f"  {e}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, preds)
    print(f"Wrote {preds.shape} z-scored predictions → {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
