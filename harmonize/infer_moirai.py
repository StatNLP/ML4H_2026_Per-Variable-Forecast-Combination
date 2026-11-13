"""
harmonize/infer_moirai.py  —  RUN WITH THE py311_moirai ENV, NOT py313_tsf.

    python -m harmonize.infer_moirai ...

Moirai (uni2ts) is a natively multivariate pretrained time-series foundation model
but pins torch<2.5 / numpy<2, so it can't share py313_tsf with Chronos + the eval.
This script is therefore **standalone** (h5py + numpy + torch + uni2ts only — it does
NOT import the harmonize package): it reads a Stage-3 test H5, forecasts every window
zero-shot, and writes the z-scored predictions to a .npy in H5 row order. The py313
eval then consumes that file via `eval_forecast --pred-file` and runs all the usual
clinical metrics (feat_mse, horizon, SOFA/ΔSOFA, AKI) — identical to the Chronos path,
so Moirai is directly comparable to Chronos-2 (both native multivariate).

Conventions (match models/chronos2.py):
- z-scored in / z-scored out (Moirai instance-norms per variate over OBSERVED steps,
  so it returns the forecast in the input's z-score space).
- Missingness is passed natively via `past_observed_target` (= mask_in) — Moirai
  ignores unobserved steps for its scaler, so the 0-filled H5 values there are inert.
  No NaN needed.
- point forecast = median over Moirai's sample paths.
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Zero-shot Moirai forecasts → .npy (run in py311_moirai)")
    ap.add_argument("--test-h5", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output .npy (N, T_out, F_target), z-scored")
    ap.add_argument("--model", default="Salesforce/moirai-1.1-R-small",
                    help="HF id: moirai-1.1-R-{small,base,large} (~14/91/311M)")
    ap.add_argument("--patch-size", default="8",
                    help="int patch size; 8 = exactly 3 patches for a 24h context (no padding). "
                         "NB Moirai 'auto' is buggy for this short fixed context — use a fixed size.")
    ap.add_argument("--num-samples", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch = args.patch_size if args.patch_size == "auto" else int(args.patch_size)

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
    ctx_all = x_in[:, :, tgt_idx]                       # (N,T_in,F)
    obs_all = mask_in[:, :, tgt_idx]                    # (N,T_in,F) bool

    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained(args.model)
    model = MoiraiForecast(
        prediction_length=T_out, target_dim=F,
        feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
        context_length=T_in, module=module, patch_size=patch,
        num_samples=args.num_samples,
    ).to(device).eval()
    print(f"Moirai {args.model} ({sum(p.numel() for p in module.parameters())/1e6:.1f}M) "
          f"on {ds_name}: {N} windows, F={F}, ctx={T_in}->{T_out}, device={device}", flush=True)

    preds = np.zeros((N, T_out, F), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, args.batch_size):
            e = min(s + args.batch_size, N)
            ctx = torch.from_numpy(ctx_all[s:e]).float().to(device)        # (B,T_in,F)
            obs = torch.from_numpy(obs_all[s:e]).to(device)                # (B,T_in,F) bool
            is_pad = torch.zeros(ctx.shape[0], T_in, dtype=torch.bool, device=device)
            samples = model(past_target=ctx, past_observed_target=obs, past_is_pad=is_pad)
            # samples: (B, num_samples, T_out, F) → median point forecast
            preds[s:e] = samples.median(dim=1).values.cpu().numpy()
            if s % (args.batch_size * 20) == 0:
                print(f"  {e}/{N}  ({time.time()-t0:.0f}s)", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, preds)
    print(f"Wrote {preds.shape} z-scored predictions → {args.out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
