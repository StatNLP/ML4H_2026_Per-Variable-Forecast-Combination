#!/usr/bin/env python3
"""Stage D — does teacher-forcing an IMS rollout with the ENSEMBLE help? (§27.10)

Post-hoc stacking can only reweight 24 already-committed trajectories. This asks a different
question: if the autoregressive decoder were fed better values *during* generation, would error
accumulation shrink? The horizon curves motivate it — error grows 2.0-2.3x from h1 to h24 and
residual correlation RISES with horizon, i.e. the models drift together.

Three variants, all on the same windows:
  (a) free      — teacher_forcing_ratio=0: the decoder consumes its own predictions (current eval)
  (b) ensemble  — the ensemble's value is injected at each step
  (c) oracle    — the ground truth is injected at each step

MASKED BY CONSTRUCTION (MF's correction). At 11-18 % observation density, injecting ground truth
everywhere would feed 82-89 % imputed values — forward-fill forcing, not an oracle. The existing
rollout already handles this: `x_target` carries NaN at unobserved positions and the loop falls
back to the model's own prediction there. So passing a surrogate target with NaN outside the mask
gives masked injection with no change to the model.

(c) is the DECISION GATE: it bounds (b), because the ensemble is at best a noisy estimate of the
truth. If the oracle barely beats free-running, injection cannot help and the idea is closed.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from harmonize.icu_dataset import ICUDataset                      # noqa: E402
from harmonize.eval_forecast import load_model                    # noqa: E402
from harmonize import featstack as FS                             # noqa: E402


def masked_mse(pred, y, m):
    d = np.where(m, pred - np.nan_to_num(y), 0.0)
    obs = m.sum(axis=(0, 1))
    per = (d ** 2).sum(axis=(0, 1)) / np.clip(obs, 1, None)
    return float(per[obs > 0].mean()), float((d ** 2).sum() / max(obs.sum(), 1))


@torch.no_grad()
def rollout(model, xi, mi, st, device, surrogate=None, bs=128):
    """surrogate=None -> free-running; else inject it (NaN outside the mask) as the AR input."""
    out = []
    for s in range(0, len(xi), bs):
        e = min(s + bs, len(xi))
        xt = None if surrogate is None else torch.from_numpy(surrogate[s:e]).to(device)
        out.append(model(torch.from_numpy(xi[s:e]).to(device),
                         torch.from_numpy(mi[s:e]).to(device),
                         torch.from_numpy(st[s:e]).to(device),
                         x_target=xt,
                         teacher_forcing_ratio=0.0 if surrogate is None else 1.0).cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    # repeatable: the ensemble/stat build is model-independent, so several IMS decoders can be
    # scored in ONE allocation off the same weights. Stage D on a single architecture cannot
    # distinguish "injection does not help" from "denseAR in particular cannot use it".
    ap.add_argument("--checkpoint", required=True, type=Path, nargs="+")
    ap.add_argument("--val-h5", required=True, type=Path)
    ap.add_argument("--pred-files", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=6000)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None, help="CSV artefact (was print-only)")
    a = ap.parse_args()

    ds = ICUDataset(str(a.val_h5), preload=True)
    import h5py
    with h5py.File(a.val_h5, "r") as f:
        n = min(a.limit, f["x_in"].shape[0])
        xi, mi = f["x_in"][:n], f["mask_in"][:n]
        st, sid = f["static"][:n], f["stay_id"][:n]
        xt, mt = f["x_target"][:n], f["mask_target"][:n].astype(bool)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ensemble over the cached pool, weights fitted on a disjoint half of the stays
    preds = []
    for spec in a.pred_files:
        nm, _, p = spec.partition("=")
        arr = np.load(p, mmap_mode="r")
        if arr.shape[0] >= n:
            preds.append(FS.Predictor(nm, "npy", arr[:n]))
    fit_idx, ev_idx = FS.stay_grouped_split(sid, 0.5)
    stats = FS.accumulate_stats(preds, fit_idx, xi, mi, st, xt, mt, device)
    W, _ = FS.build_weight_sets(stats, 1.0, 200)
    w = W["simplex_var"][:, 0, :]                       # (F, M), hour-tied
    P = np.stack([np.asarray(p.obj) for p in preds], axis=-1)      # (n,T,F,M)
    ens = np.einsum("nhfm,fm->nhf", P, w).astype(np.float32)

    # --- surrogates -------------------------------------------------------------------------
    # MASKED variants inject only where the target is observed and fall back to the model's own
    # prediction elsewhere. That is forced for GROUND TRUTH (it exists nowhere else) but it hands
    # the decoder a jagged mix — real values at 11-18 % of cells, its own smooth output at the
    # rest — which is a distribution shift for a decoder trained by scheduled sampling on its own
    # smooth rollouts. That shift, not the injected values, is what made the masked oracle lose.
    #
    # A MODEL's prediction exists at EVERY cell, so injecting one needs no mask: the decoder gets
    # a complete, smooth trajectory that is simply better than its own. This is MF's variant and
    # the masked oracle does NOT bound it.
    sur_ens_masked = np.where(mt, ens, np.nan).astype(np.float32)
    sur_gt_masked = np.where(mt, np.nan_to_num(xt), np.nan).astype(np.float32)
    sur_ens_full = ens.astype(np.float32)                      # unmasked: every cell
    # per-variable best single model (fit half), injected everywhere — denseAR keeps its own
    # channels only where it is already the best, and hands e.g. medications to whoever wins them
    per_var = stats["sse"].sum(axis=1) / np.clip(stats["n"].sum(axis=1), 1, None)[:, None]
    win = per_var.argmin(axis=1)                                # (F,) best model per variable
    sur_best = np.stack([np.asarray(preds[win[j]].obj)[:, :, j] for j in range(ens.shape[2])],
                        axis=-1).astype(np.float32)
    nm = [preds[i].name for i in win]
    print(f"per-variable winners span {len(set(nm))} distinct models")

    # ---- per-checkpoint loop: everything above is shared across architectures ----------------
    # Stage D on a single architecture cannot distinguish "injection does not help" from
    # "denseAR in particular cannot use it", so the driver scores every IMS decoder given.
    all_rows = []
    for ckpt in a.checkpoint:
        model, cfg = load_model(ckpt, ds, device); model.eval()

        # (f) MF's exact variant: at each step the decoder keeps its OWN LIVE output on the
        # channels it wins, and receives the winning model's line on the channels it loses. NaN
        # is the signal for "use your own step_pred", so marking self-won channels NaN gives the
        # live hybrid — unlike (e), which substituted the decoder's CACHED free-running values
        # there and therefore could not react to the trajectory changing underneath it.
        own = ckpt.name
        own_idx = next((i for i, pp in enumerate(preds) if pp.name == own), None)
        if own_idx is None:  # labels are free-form; a mismatch would make (f) a copy of (e)
            own_idx = next((i for i, pp in enumerate(preds)
                            if own in pp.name or pp.name in own), None)
        if own_idx is None:
            raise SystemExit(f"cannot locate {own!r} among pool labels "
                             f"{[pp.name for pp in preds]} — variant (f) would duplicate (e)")
        sur_hybrid = sur_best.copy()
        keep = np.flatnonzero(win == own_idx)
        sur_hybrid[:, :, keep] = np.nan
        print(f"hybrid: {ckpt.name} keeps {len(keep)}/{ens.shape[2]} channels live, "
              f"{ens.shape[2]-len(keep)} injected from other models")
        # Score on the EVAL half only. `ev_idx` was computed and then discarded, so every variant
        # was measured on all windows — including the half the combiner weights were fitted on,
        # which flatters every injected variant relative to free-running.
        def sub(arr):
          return None if arr is None else arr[ev_idx]

        rows = []
        for tag, sur in (("(a) free-running", None),
                       ("(b) ensemble, masked", sur_ens_masked),
                       ("(c) oracle, masked", sur_gt_masked),
                       ("(d) ensemble, FULL", sur_ens_full),
                       ("(e) per-var best, FULL", sur_best),
                       ("(f) live hybrid (MF)", sur_hybrid)):
          pred_ev = rollout(model, xi[ev_idx], mi[ev_idx], st[ev_idx], device, sub(sur),
                            a.batch_size)
          pv, gl = masked_mse(pred_ev, xt[ev_idx], mt[ev_idx])
          rows.append((tag, pv, gl))
        base = rows[0][1]
        print(f"\nStage D — {ckpt.name} on {ds.split}, {len(ev_idx)} EVAL-half windows "
            f"(of {n}), {len(preds)} pool models, weights fitted on the disjoint half\n")
        print(f"{'variant':22s} {'mse_per_var':>12} {'mse_global':>11} {'vs free':>9}")
        for tag, pv, gl in rows:
          print(f"{tag:22s} {pv:12.5f} {gl:11.5f} {100*(pv-base)/base:8.2f}%")
          all_rows.append((ckpt.name, tag, pv, gl, 100*(pv-base)/base))
        print(f"\nNB the masked oracle (c) bounds only the MASKED ensemble (b). The FULL variants "
            f"(d,e) inject a complete smooth trajectory and are not bounded by it.")

    # persist — these numbers were print-only, so no Stage D artefact survived the logs
    if a.out:
        import csv
        a.out.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w", newline="") as fh:
            w_ = csv.writer(fh)
            w_.writerow(["checkpoint", "variant", "mse_per_var", "mse_global", "vs_free_pct"])
            w_.writerows(all_rows)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
