"""
harmonize/ensemble_eval.py
Stage 5b — multi-model ensemble + orthogonality analysis.

Loads several trained checkpoints (ideally different model families:
dense_autoregressive / dense_dms / informer / autoformer / dlinear), runs each
on the test split, and asks two questions:

  1. ORTHOGONALITY — how decorrelated are the per-model forecast residuals?
     Low mean pairwise residual correlation ⇒ high ensemble-gain potential
     (ensemble error variance reduction ∝ 1 − mean ρ).

  2. DOES THE ENSEMBLE HELP? — builds a mean ensemble (and an inverse-MSE
     skill-weighted ensemble) of the predictions, runs the SAME downstream
     clinical scores (forecast MSE, SOFA, ΔSOFA≥2 detection, SAPS-II, optional
     Sepsis-3), and compares ensemble vs. the best single model on BOTH axes:
       - aggregate forecast/SOFA error  (where averaging should help)
       - ΔSOFA≥2 deterioration sensitivity (where averaging — variance
         reduction — may NOT help and can hurt, because it regresses the rare
         spikes further toward the mean). This contrast is the point.

Memory-efficient: processes one model at a time, folds its prediction into a
running ensemble sum, and keeps only a residual subsample for the correlation
matrix.

Usage:
    python -m harmonize.ensemble_eval \
        --checkpoints checkpoints/umm__4h_d128_l1_dr0_2_s42 \
                      checkpoints/dense_dms_umm_4h_d128 \
                      checkpoints/informer_umm_4h_d128 \
        --test-h5 data/dataset/umm_seed42_test__all_4h_fwd.h5 \
        [--long-parquet data/long/umm_long.parquet] \
        [--out docs/tables/ensemble_umm.json]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from harmonize.icu_dataset import ICUDataset
from harmonize.eval_forecast import load_model, sofa_comparison, sapsii_comparison
from harmonize.scores.sofa import denormalise, locf

log = logging.getLogger("ensemble_eval")
ROOT = Path(__file__).resolve().parents[1]


def gather_tensors(ds: ICUDataset, batch_size: int, limit: int | None):
    """One pass over the dataset → numpy arrays (model-independent)."""
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    xi, mi, st, xt, mt, sid, t0 = [], [], [], [], [], [], []
    seen = 0
    for b in loader:
        if limit and seen >= limit:
            break
        seen += b["x_in"].shape[0]
        xi.append(b["x_in"].numpy()); mi.append(b["mask_in"].numpy())
        st.append(b["static"].numpy()); xt.append(b["x_target"].numpy())
        mt.append(b["mask_target"].numpy()); sid.append(b["stay_id"].numpy())
        t0.append(b["t0_hour"].numpy())          # Sepsis-3 ±24 h coincidence (§37.38 D19)
    return (np.concatenate(xi), np.concatenate(mi), np.concatenate(st),
            np.concatenate(xt), np.concatenate(mt), np.concatenate(sid),
            np.concatenate(t0))


@torch.no_grad()
def predict(model, x_in, mask_in, static, device, batch_size=256):
    """Batched forward → predictions in z-score space (N, T_out, F_target)."""
    out = []
    for s in range(0, len(x_in), batch_size):
        e = s + batch_size
        out.append(model(
            torch.from_numpy(x_in[s:e]).to(device),
            torch.from_numpy(mask_in[s:e]).to(device),
            torch.from_numpy(static[s:e]).to(device),
        ).cpu().numpy())
    return np.concatenate(out)


def forecast_err(pred_z, xt_z, mt):
    """Masked per-variable + global MSE/MAE in z-score space."""
    diff = np.where(mt, pred_z - np.nan_to_num(xt_z), 0.0)
    obs = mt.sum(axis=(0, 1)); denom = np.clip(obs, 1, None)
    return {
        "mse_per_var": float(np.mean((diff**2).sum(axis=(0, 1)) / denom)),
        "mae_per_var": float(np.mean(np.abs(diff).sum(axis=(0, 1)) / denom)),
        "mse":         float((diff**2).sum() / obs.sum()),
    }


def clinical_scores(pred_z, gt_phys, feats_tgt, dataset, static, feats_static, stay_id=None):
    """SOFA + SAPS-II comparison for a denormalised prediction vs ground truth."""
    pred_phys = denormalise(pred_z, feats_tgt, NORM)
    sofa_df, sofa_m = sofa_comparison(pred_phys, gt_phys, feats_tgt, dataset)
    saps_m = (sapsii_comparison(pred_phys, gt_phys, feats_tgt, static, feats_static, dataset, stay_id)
              if static.shape[1] > 0 else {})
    return sofa_df, sofa_m, saps_m


NORM: pd.DataFrame = None   # set in main (module-level for clinical_scores reuse)
TABLE_TAG = ""             # --table-tag, set in main; keeps robustness pools off published tables


def stays_per_channel(mask_target, sid) -> "np.ndarray":
    """Distinct stays contributing at least one observed cell, per channel -> (F,).

    BMA's effective sample size must be per channel. Using one database-wide stay count gives a
    channel seen in a dozen stays the same N as one seen in thousands, so its posterior collapses
    onto a single member -- the very failure the stay-level correction was introduced to rule out.
    """
    m = np.asarray(mask_target).astype(bool)          # (N, T, F)
    seen = m.any(axis=1)                              # (N, F): stay-window touched the channel
    s = np.asarray(sid)
    _, inv = np.unique(s, return_inverse=True)
    F = seen.shape[1]
    out = np.zeros(F, dtype=float)
    for f in range(F):
        out[f] = len(np.unique(inv[seen[:, f]])) if seen[:, f].any() else 0.0
    return np.clip(out, 1.0, None)


def _run_featstack(args, ds, device, xi, mi, st, xt, mt, sid, feats_tgt, dataset) -> None:
    """Per-(variable, hour) stacking with a stay-grouped fit/eval split (§13.6b).

    The whole point is that the combiners are FITTED and SCORED on disjoint patients, which is
    what §13.6a lacked — its per-variable oracle was fitted and scored on the same windows, so
    its +12-24 % 'gain' was selection noise. Anything reported here has to beat `best_single`
    on stays the combiner never saw.
    """
    from harmonize import featstack as FS

    # GUARD: the combiners are FITTED here, so running this on `test` spends the final held-out
    # set — after which no honest confirmation is possible without building a new one. Fit on
    # `val` (stay-grouped into fit/eval halves) and keep test for a single last confirmation.
    split = str(getattr(ds, "split", "") or "unknown")
    # `--dump-preds` is exempt: caching predictions is pure inference and SELECTS nothing, so it
    # does not spend the held-out set — and the final confirmation needs exactly those test
    # predictions to exist. Only FITTING combiners on test is forbidden.
    # `--dump-combined WITH --combined-weights-from` is also exempt: the weights were fitted
    # elsewhere and are only being APPLIED here, so nothing is selected on test and the held-out
    # set is not spent. `--dump-combined` WITHOUT stored weights would fit on test and stays
    # forbidden — which is what this guard caught when ENSTRAJ first ran.
    _applying_stored_weights = bool(args.dump_combined and args.combined_weights_from)
    if (split == "test" and not args.allow_test_fit and not args.dump_preds
            and not _applying_stored_weights):
        raise SystemExit(
            "refusing to fit combiners on the TEST split (this would burn the final held-out "
            "set). Pass --test-h5 <...>_val__... instead, or --allow-test-fit for the one "
            "deliberate final confirmation. (Caching predictions with --dump-preds is allowed, "
            "and so is --dump-combined when --combined-weights-from supplies the weights, "
            "because then nothing is fitted here.)")

    # --- optional row filter: admission windows only -------------------------------------
    # The outcome heads (§16.10a) train with `--first-day-only`, i.e. on t0_hour == 0 only. Dumping
    # or combining over the full split would materialise ~23x more data than those heads can use
    # (eICU train is 6.2 GB per member at full width, 270 MB at admission width). `t0_hour == 0`
    # rows are selected in ascending dataset order here and in `train_head._admission`, so row i of
    # a `_fd` file is row i of the head's subset. The split tag gains `_fd` so a filtered dump can
    # never be mistaken for a full one.
    if getattr(args, "rows", "all") == "first-day":
        import h5py
        with h5py.File(args.test_h5, "r") as _f:
            sel = np.flatnonzero(_f["t0_hour"][:] == 0)
        if sel.size == 0:
            raise SystemExit("--rows first-day selected 0 windows; is t0_hour present?")
        log.info("--rows first-day: %d -> %d windows", len(xi), sel.size)
        xi, mi, st = xi[sel], mi[sel], st[sel]
        xt, mt, sid = xt[sel], mt[sel], sid[sel]
        split = f"{split}_fd"

    preds = []
    for ck in args.checkpoints:
        model, cfg = load_model(ck, ds, device)
        model.eval()
        preds.append(FS.Predictor(cfg.get("run_name", ck.name), "torch", model))
    for spec in args.pred_files:
        name, _, path = spec.partition("=")
        arr = np.load(path, mmap_mode="r")
        if arr.shape[0] == len(xi):
            pass
        elif args.limit and arr.shape[0] > len(xi):
            # `--limit N` truncates the dataset to its first N windows; predictions are stored in
            # dataset row order, so the matching rows are exactly the first N. Without this every
            # cached file was rejected on the row check and the pool silently became empty.
            arr = arr[:len(xi)]
        else:
            log.warning("skip %s: %d rows != %d windows", name, arr.shape[0], len(xi))
            continue
        preds.append(FS.Predictor(name, "npy", arr))
    log.info("featstack base models (%d): %s", len(preds), ", ".join(p.name for p in preds))

    # --- optional: cache every base model's predictions, then exit -------------------------
    # Inference is the only expensive step here; the combiners themselves are closed-form numpy
    # over sufficient statistics. Dumping predictions once turns every subsequent experiment
    # (new combiner, cluster count, ridge lambda, pool subset) into a CPU job of seconds with no
    # checkpoint loading at all — re-run with `--checkpoints` omitted and these files passed as
    # `--pred-files`. Small models on CPU are perfectly adequate for this: measured 71 win/s
    # (denseAR), 11 401 win/s (DLinear), 32 win/s (STraTS) on 8 threads.
    if args.dump_preds:
        out_dir = Path(args.dump_preds); out_dir.mkdir(parents=True, exist_ok=True)
        for p in preds:
            if p.kind == "npy":
                continue                       # already on disk
            dest = out_dir / f"{p.name}__{dataset}_{split}.npy"
            if dest.exists():
                log.info("skip %s (exists)", dest.name); continue
            buf = np.empty((len(xi), xt.shape[1], len(feats_tgt)), dtype=np.float32)
            for s in range(0, len(xi), 1024):
                idx = np.arange(s, min(s + 1024, len(xi)))
                buf[idx] = p.predict(idx, xi, mi, st, device)
            np.save(dest, buf)
            log.info("wrote %s  %s", dest.name, buf.shape)
        print(f"dumped {sum(1 for p in preds if p.kind != 'npy')} prediction files -> {out_dir}")
        return

    # Ensembling (but NOT dumping, which is one model per job by design) needs >= 2 models.
    if len(preds) < 2:
        raise SystemExit(
            f"only {len(preds)} base model(s) available — an ensemble needs at least 2. "
            "Check that --pred-files row counts match the (possibly --limit-ed) split.")

    # --- FINAL CONFIRMATION PATH: fit on ALL of val, evaluate on ALL of test ----------------
    # The stay-grouped half-split below is the development protocol; it exists so combiners can
    # be compared repeatedly without touching test. The final run is different and happens ONCE:
    # every val window fits the weights, every test window scores them. Nothing is selected on
    # test — the combiner is already chosen before this runs.
    if args.eval_h5:
        if split == "test":
            raise SystemExit("--test-h5 should be the VAL split (the fitting data) and "
                             "--eval-h5 the test split")
        ds_ev = ICUDataset(str(args.eval_h5), preload=True)
        xi2, mi2, st2, xt2, mt2, sid2, _t02 = gather_tensors(ds_ev, args.batch_size, None)
        ev_preds = []
        for spec in args.eval_pred_files:
            name, _, path = spec.partition("=")
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] != len(xi2):
                raise SystemExit(f"{name}: {arr.shape[0]} rows != {len(xi2)} eval windows")
            ev_preds.append(FS.Predictor(name, "npy", arr))
        # Weight vectors are indexed by POSITION, so the two splits must list the same models in
        # the same order — otherwise every weight is silently applied to the wrong model.
        if [p.name for p in ev_preds] != [p.name for p in preds]:
            raise SystemExit("fit/eval model lists differ (order matters):\n"
                             f"  fit : {[p.name for p in preds]}\n"
                             f"  eval: {[p.name for p in ev_preds]}")
        log.info("FINAL: fitting on %d %s windows, scoring on %d %s windows",
                 len(xi), split, len(xi2), getattr(ds_ev, "split", "?"))
        s_fit = FS.accumulate_stats(preds, np.arange(len(xi)), xi, mi, st, xt, mt, device)
        # BMA's effective sample size is the number of STAYS, not observed cells: cells within a
        # stay are far from independent, and using the cell count collapses the posterior onto one
        # model on every channel.
        W, diag = FS.build_weight_sets(s_fit, args.ridge_lambda, args.min_support,
                                       n_eff=stays_per_channel(mt, sid),
                                       sweep=args.combiner_sweep)
        s_ev = FS.accumulate_stats(ev_preds, np.arange(len(xi2)), xi2, mi2, st2, xt2, mt2, device)
        scores = FS.score_weights(W, s_ev)
        base = scores["best_single"]["mse_per_var"]
        rows = [{"dataset": dataset, "fitted_on": split,
                 "evaluated_on": str(getattr(ds_ev, "split", "?")), "combiner": k,
                 "mse_per_var": round(v["mse_per_var"], 5),
                 "mse_global": round(v["mse_global"], 5),
                 "delta_vs_best_single_pct": round(100.0 * (v["mse_per_var"] - base) / base, 2)}
                for k, v in sorted(scores.items(), key=lambda kv: kv[1]["mse_per_var"])]
        df = pd.DataFrame(rows)
        print(f"\n=== FINAL {dataset}: fit on {split}, evaluated on "
              f"{getattr(ds_ev, 'split', '?')} ===")
        print(f"best single base model on {split}: {preds[diag['best_m']].name}")
        print(df.to_string(index=False))
        # A tagged FINAL run (rebuttal pools) writes into its own folder, so it can never overwrite
        # the published FINAL tables or any appendix table derived from them.
        tdir = _final_dir()
        df.to_csv(tdir / f"featstack_FINAL_{dataset}.csv", index=False)
        print(f"wrote {tdir}/featstack_FINAL_{dataset}.csv")

        if args.final_extras:
            final_extras(dataset, preds, ev_preds, W, diag, xt2, mt2, sid2,
                         stats_ev=s_ev, tdir=tdir, feature_names=feats_tgt,
                         n_perm=args.n_perm, seed=args.split_seed)
        if args.final_support:
            final_support(dataset, ev_preds, W, diag, xt2, mt2, sid2, args.eval_h5, feats_tgt,
                          val_n=np.asarray(mt).sum(axis=(0, 1)))
        if args.final_clinical:
            final_clinical(dataset, ev_preds, W, diag, ds_ev, xi2, xt2, mt2, st2, sid2, tdir)
        return

    if args.final_extras:
        raise SystemExit("--final-extras only applies to the FINAL path; pass --eval-h5 (test) "
                         "with --test-h5 (val), or drop the flag.")

    fit_idx, ev_idx = FS.stay_grouped_split(sid, args.fit_frac, args.split_seed)
    log.info("stay-grouped split: %d fit windows / %d eval windows (%d/%d stays)",
             len(fit_idx), len(ev_idx), len(np.unique(sid[fit_idx])), len(np.unique(sid[ev_idx])))

    s_fit = FS.accumulate_stats(preds, fit_idx, xi, mi, st, xt, mt, device)
    W, diag = FS.build_weight_sets(s_fit, args.ridge_lambda, args.min_support,
                                   n_eff=stays_per_channel(mt[fit_idx], sid[fit_idx]),
                                   sweep=args.combiner_sweep)
    weight_concentration(dataset, W, s_fit)
    member_weights(dataset, W, s_fit, [pr.name for pr in preds])

    # --- optional: materialise the COMBINED trajectory and exit ----------------------------
    # score_weights works from sufficient statistics and never builds the combined prediction, so
    # anything that wants the ensemble's actual forecast as a tensor (the §16.10a outcome heads)
    # needs it written out. comb[n,h,f] = sum_m W[f,h,m] * P[n,h,f,m].
    #
    # The weights are fitted on the FIT HALF of the split passed here, which must therefore be
    # `val` — the same discipline as every other combiner in this module. The emitted file covers
    # ALL rows of the split, because a head trained downstream needs the whole thing; the
    # fit/eval provenance is recorded in the sidecar so a later reader cannot forget that the
    # val rows in it were partly used to fit the weights.
    if args.dump_combined:
        name = args.combined_weight_set
        if name not in W:
            raise SystemExit(f"--combined-weight-set {name!r} not built; available: "
                             f"{', '.join(sorted(W))}")
        out_dir = Path(args.dump_combined); out_dir.mkdir(parents=True, exist_ok=True)
        w = np.asarray(W[name], dtype=np.float32)            # (F, H, M)
        # The ensemble must be ONE fixed function applied to every split, not refit per split:
        # a head that trains on train-combined and is tested on test-combined has to see the same
        # combiner on both sides, or the arms differ by more than the data. So the weights are
        # fitted once (on whichever split is passed first, by convention train) and reused.
        wpath = out_dir / f"weights_{name}__{dataset}.npz"
        if args.combined_weights_from:
            z = np.load(args.combined_weights_from, allow_pickle=True)
            if list(z["members"]) != [pr.name for pr in preds]:
                raise SystemExit(
                    f"{Path(args.combined_weights_from).name} was fitted on a different pool:\n"
                    f"  stored: {list(z['members'])}\n  here:   {[pr.name for pr in preds]}")
            w = z["w"].astype(np.float32)
            log.info("reusing weights fitted on %s (%s)", z["fitted_on_split"], z["fitted_on_dataset"])
        else:
            np.savez(wpath, w=w, members=np.array([pr.name for pr in preds]),
                     weight_set=name, fitted_on_dataset=dataset, fitted_on_split=split,
                     n_fit_windows=len(fit_idx))
            log.info("wrote %s — pass it as --combined-weights-from for the other splits", wpath.name)
        comb = np.empty((len(xi), xt.shape[1], len(feats_tgt)), dtype=np.float32)
        for s in range(0, len(xi), 512):
            idx = np.arange(s, min(s + 512, len(xi)))
            P = np.stack([pr.predict(idx, xi, mi, st, device) for pr in preds], axis=-1)
            comb[idx] = np.einsum("fhm,nhfm->nhf", w, P, optimize=True)
        dest = out_dir / f"ensemble_{name}__{dataset}_{split}.npy"
        np.save(dest, comb)
        # Provenance must say where the weights ACTUALLY came from. Writing "fit half of this
        # split" unconditionally was wrong for every reused-weights file -- i.e. for val and test,
        # which is 10 of 15 -- and that is precisely the question this sidecar exists to answer.
        if args.combined_weights_from:
            prov = {"weights_fitted_on": f"{str(z['fitted_on_split'])} of "
                                         f"{str(z['fitted_on_dataset'])} (reused)",
                    "weights_file": str(args.combined_weights_from),
                    "n_fit_windows": int(z["n_fit_windows"])}
        else:
            prov = {"weights_fitted_on": f"fit half of {split} (fitted here)",
                    "weights_file": str(wpath),
                    "n_fit_windows": int(len(fit_idx)), "n_eval_windows": int(len(ev_idx))}
        (dest.with_suffix(".json")).write_text(json.dumps({
            "weight_set": name, "dataset": dataset, "split": split,
            "rows": int(len(xi)), "members": [pr.name for pr in preds],
            **prov,
            "fit_frac": args.fit_frac, "split_seed": args.split_seed,
            "row_filter": getattr(args, "rows", "all"),
        }, indent=2))
        log.info("wrote %s  %s", dest.name, comb.shape)
        print(f"combined trajectory -> {dest}")
        return

    s_ev = FS.accumulate_stats(preds, ev_idx, xi, mi, st, xt, mt, device)
    scores = FS.score_weights(W, s_ev)
    if args.rebuttal_extras:
        _rebuttal_dev_extras(dataset, preds, W, s_ev, ev_idx, xi, mi, st, xt, mt, sid, feats_tgt,
                             device, args)

    # --- Tier 4: patient clusters x ranking-only weights (models trained on ALL data) ---
    if args.n_clusters > 1:
        try:
            import h5py
            with h5py.File(args.test_h5, "r") as h5:
                sofa = h5["sofa_scores"][:len(xi)] if "sofa_scores" in h5 else None
                t0 = h5["t0_hour"][:len(xi)] if "t0_hour" in h5 else None
            if args.cluster_mode == "physio":
                # observed measurements only — no statics (their missingness produced the
                # "weight 23.2 / height 0.0" pseudo-cluster), no density, no channel count
                meta, meta_stats, _ = FS.physio_features(xi, mi, sofa, args.cluster_top_n)
            elif args.cluster_mode == "error":
                # ORACLE upper bound: clusters built from the per-model error profile. Uses the
                # targets, so it is not deployable — it bounds what any input-based router could
                # achieve. Fitted on the fit half and applied to the eval half like the others.
                # Standardise ONCE, on the fit half, and apply those statistics to both halves.
                # Scoring each half with its own mu/sd put the two in different units, so KMeans
                # centroids learned on the fit half were applied to eval-half coordinates.
                meta_all, (mu, sd, nmz) = FS.error_features(
                    preds, np.arange(len(xi)), xi, mi, st, xt, mt, device, raw=True)
                meta = ((meta_all - mu) / sd).astype(np.float32)
                meta_stats = (mu, sd, nmz)
            elif args.cluster_mode == "router":
                # Deployable two-stage router: cluster on the errors of the FIT half, then assign
                # eval-half windows from their INPUT alone. This is the honest counterpart to the
                # `error` ceiling — no target of an eval window touches its routing decision.
                err_f, _ = FS.error_features(preds, fit_idx, xi, mi, st, xt, mt, device)
                ph, ph_stats, _ = FS.physio_features(xi, mi, sofa, args.cluster_top_n)
                c_fit, c_ev, rdiag = FS.fit_router(err_f, ph[fit_idx], ph[ev_idx], args.n_clusters)
                meta, meta_stats, km = ph, ph_stats, None
                log.info("router: fit sizes %s -> predicted eval sizes %s (train acc %.3f)",
                         rdiag["fit_sizes"], rdiag["pred_sizes"], rdiag["train_acc"])
            else:
                meta, meta_stats = FS.window_meta_features(xi, mi, st, sofa, t0)
            if args.cluster_mode != "router":
                km = FS.fit_clusters(meta[fit_idx], args.n_clusters)
                c_fit, c_ev = km.predict(meta[fit_idx]), km.predict(meta[ev_idx])
            sizes = np.bincount(c_fit, minlength=args.n_clusters)
            log.info("clusters (fit half): %s", dict(enumerate(sizes.tolist())))

            cs_fit = FS.accumulate_cluster_stats(preds, fit_idx, xi, mi, st, xt, mt,
                                                 c_fit, args.n_clusters, device)
            cs_ev = FS.accumulate_cluster_stats(preds, ev_idx, xi, mi, st, xt, mt,
                                                c_ev, args.n_clusters, device)
            # MF's two-stage test: when the clusters are defined on the TARGET side (error mode),
            # go back to the INPUT and ask whether they are recognisable at all. If not, no router
            # can route — "is there any context or not".
            if args.cluster_mode == "error":
                inp, inp_stats = FS.window_meta_features(xi, mi, st, sofa, t0)
                ph, ph_stats, _ = FS.physio_features(xi, mi, sofa, args.cluster_top_n)
                for tag, X, S in (("meta", inp, inp_stats), ("physio", ph, ph_stats)):
                    ctx = FS.cluster_context_test(c_fit, c_ev, X[fit_idx], X[ev_idx], S,
                                                  args.n_clusters)
                    print(f"\n--- CONTEXT TEST ({tag} inputs -> error-defined clusters, k={args.n_clusters}) ---")
                    print(f"  balanced accuracy {ctx['balanced_accuracy']:.3f}  vs chance "
                          f"{ctx['chance']:.3f}  -> lift {ctx['balanced_accuracy']/ctx['chance']:.2f}x")
                    print("  most informative input features: "
                          + ", ".join(f"{n}({v:.2f})" for n, v in ctx["top_features"]))

            # mechanistic diagnostic: do the clusters actually change the winning architecture?
            desc = (FS.describe_clusters(km, meta_stats, cs_fit, [p.name for p in preds])
                    if km is not None else None)
            if desc is None:
                per = cs_fit["sse"].sum(axis=1) / np.clip(cs_fit["n"].sum(axis=1), 1, None)[:, None]
                wins = [preds[int(np.argmin(per[k]))].name for k in range(args.n_clusters)]
                print(f"\n--- router clusters (k={args.n_clusters}) best architecture per cluster ---")
                for k, w in enumerate(wins):
                    print(f"  cluster {k}: {w}")
                print(f"  distinct winning architectures across clusters: {len(set(wins))}")
                desc = {"winners": wins}
            else:
                print(f"\n--- cluster prototypes (k={args.n_clusters}, fitted on the fit half) ---")
            if desc.get("names") is not None:
                hdr = "  ".join(f"{n[:13]:>13}" for n in desc["names"])
                print(f"{'cl':>3} {'n':>6}  {hdr}   best architecture")
                for k in range(args.n_clusters):
                    vals = "  ".join(f"{v:>13.2f}" for v in desc["centroids"][k])
                    print(f"{k:>3} {int(sizes[k]):>6}  {vals}   {desc['winners'][k]}")
                uniq = {w for w in desc["winners"] if w}
                print(f"  distinct winning architectures across clusters: {len(uniq)} -> {sorted(uniq)}")

            gvar = W["rank_harmonic_top5"][:, 0, :]          # global fallback for thin cells
            for scheme in ("harmonic", "inv_mse"):
                Wc = FS.cluster_ranking_weights(cs_fit, gvar, args.min_support, scheme=scheme)
                sc = FS.score_cluster_weights(Wc, cs_ev)
                scores[f"cluster_{scheme}_k{args.n_clusters}"] = dict(
                    mse_per_var=sc["mse_per_var"], mse_global=sc["mse_global"],
                    per_feature=sc["per_feature"], evaluable_features=int((cs_ev["n"].sum(0) > 0).sum()))
            # control: the SAME global weights scored through the cluster path — isolates the
            # effect of clustering from any difference in the scoring route.
            sc0 = FS.score_cluster_weights(
                np.broadcast_to(gvar, (args.n_clusters,) + gvar.shape).copy(), cs_ev)
            scores["cluster_control_global"] = dict(
                mse_per_var=sc0["mse_per_var"], mse_global=sc0["mse_global"],
                per_feature=sc0["per_feature"], evaluable_features=0)
        except Exception as e:
            # log the traceback: a bare message turned a NameError in the PRINTING code into a
            # silent "tier skipped", and the run still reported a full-looking ladder.
            log.exception("cluster tier FAILED (not skipped) — results below omit it: %s", e)

    # --- diversity: how much can ANY combiner buy, before fitting one? (§13.6b) ---
    amb = FS.ambiguity_decomposition(s_ev)
    rc = FS.residual_correlation(s_ev)
    wcell = s_ev["n"] / max(s_ev["n"].sum(), 1)           # weight cells by observation count
    # NB: do NOT call these `mi`/`st` — those names are the mask_in / static parameters.
    mean_ind = float((amb["mean_individual"] * wcell).sum())
    ambig = float((amb["ambiguity"] * wcell).sum())
    rho = float((rc["mean_pairwise"] * wcell).sum())
    # Persist the full M x M matrix, observation-weighted over cells, so the pool's diversity
    # structure can be reported rather than summarised by one scalar. Weighting by cell counts
    # matches how rho above is formed, so the matrix and the scalar are the same quantity.
    try:
        _w = wcell[..., None, None]
        _pair = (rc["full"] * _w).sum(axis=(0, 1)) / max(float(wcell.sum()), 1e-12)
        _short = [p.name for p in preds]
        _tdir = ROOT / "docs" / "tables"; _tdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(_pair, index=_short, columns=_short).round(4)\
          .to_csv(_tdir / f"residual_correlation_{dataset}{args.table_tag or ''}.csv")
        print(f"  wrote {_tdir}/residual_correlation_{dataset}{args.table_tag or ''}.csv")
    except Exception as e:                                  # diagnostics must not kill a run
        log.warning("residual-correlation matrix not written: %s", e)
    print(f"\n--- diversity (eval half, observation-weighted) ---")
    print(f"  mean individual MSE     {mean_ind:.4f}")
    print(f"  ambiguity (diversity)   {ambig:.4f}   -> uniform-ensemble MSE "
          f"{mean_ind - ambig:.4f} ({-100.0 * ambig / mean_ind:.1f}% vs mean individual)")
    print(f"  mean pairwise residual correlation {rho:.3f}  (§13.5 global was 0.94-0.97)")
    # Consistency check: the decomposition is an identity, so the predicted uniform-ensemble
    # error must equal the `mean` combiner scored through the completely separate weight path.
    # A mismatch means one of the two is wrong — better to fail loudly than to publish either.
    got = scores["mean"]["mse_global"]
    if abs((mean_ind - ambig) - got) > 1e-6 * max(1.0, abs(got)):
        log.warning("ambiguity identity mismatch: predicted %.8f vs scored %.8f",
                    mean_ind - ambig, got)
    else:
        log.info("ambiguity identity checks out against the scored mean ensemble (%.6f)", got)

    base = scores["best_single"]["mse_per_var"]
    rows = []
    for k, v in sorted(scores.items(), key=lambda kv: kv[1]["mse_per_var"]):
        rows.append({"dataset": dataset, "combiner": k,
                     "mse_per_var": round(v["mse_per_var"], 5),
                     "mse_global": round(v["mse_global"], 5),
                     "delta_vs_best_single_pct": round(100.0 * (v["mse_per_var"] - base) / base, 2)})
    df = pd.DataFrame(rows)

    # --- grouped reporting: mse_per_var averaged over OBSERVED members of each clinical group ---
    grp = FS.feature_groups(feats_tgt, s_ev["n"])
    obs_per_feat = s_ev["n"].sum(axis=1)
    # `mse_per_var` weights every channel equally, so a channel with a handful of observations
    # counts as much as one with 165k. On HiRID `vasopressin` has 12 observed cells in the whole
    # val split and contributed 41 % of the medication group mean. Report the equal-weight number
    # only over channels clearing a support floor, and give the observation-weighted number
    # alongside so the two cannot be confused.
    MIN_OBS = args.group_min_obs
    grows = []
    for gname, mask in grp.items():
        if mask.sum() == 0:
            continue
        keep = mask & (obs_per_feat >= MIN_OBS)
        if keep.sum() == 0:
            keep = mask
        r = {"dataset": dataset, "group": gname, "n_vars": int(mask.sum()),
             "n_vars_kept": int(keep.sum()), "min_obs": MIN_OBS}
        for cname, v in scores.items():
            pf = np.asarray(v["per_feature"])
            r[cname] = round(float(pf[keep].mean()), 5)
            r[f"{cname}__obsw"] = round(
                float((pf[mask] * obs_per_feat[mask]).sum() / max(obs_per_feat[mask].sum(), 1)), 5)
        grows.append(r)
    gdf = pd.DataFrame(grows)

    print(f"\n=== featstack {dataset} [fitted on {split}] — eval half (held-out stays) ===")
    print(f"best single base model on fit half: {preds[diag['best_m']].name}")
    print(df.to_string(index=False))

    # Tag every artefact with the split it was FITTED on. The combiners must never be fitted on
    # test: test is the single final held-out confirmation and is spent the moment it is used to
    # choose weights. Fit/eval are disjoint STAY groups inside whatever split is passed here.
    tdir = ROOT / "docs" / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    df.insert(1, "fitted_on", split)
    # include the split seed in the filename: the seed sweep silently overwrote the canonical
    # seed-42 CSV and _winners.npz with whichever seed ran last
    tag = ("" if args.split_seed == 42 else f"_seed{args.split_seed}") + (args.table_tag or "")
    df.to_csv(tdir / f"featstack_{dataset}_{split}{tag}.csv", index=False)
    if not gdf.empty:
        gdf.insert(1, "fitted_on", split)
        gdf.to_csv(tdir / f"featstack_{dataset}_{split}{tag}_by_group.csv", index=False)
        keep = [c for c in ("best_single", "inv_mse_var", "rank_harmonic_top5", "simplex_var",
                            "greedy_var", "select_var", "ridge_var") if c in gdf.columns]
        print(f"\n--- mse_per_var by clinical group (equal weight over channels with "
              f">= {MIN_OBS} observed cells) ---")
        print(gdf[["group", "n_vars", "n_vars_kept"] + keep].to_string(index=False))
    # winner map = "when is which model better" (variable x hour argmin on the fit half)
    win = np.asarray(diag["cell_mse"]).argmin(axis=2)
    np.savez_compressed(tdir / f"featstack_{dataset}_{split}{tag}_winners.npz",
                        winners=win, n=s_fit["n"], features=np.array(feats_tgt),
                        models=np.array([p.name for p in preds]),
                        per_var=diag["per_var"], global_mse=diag["global_mse"],
                        ambiguity=amb["ambiguity"], mean_individual=amb["mean_individual"],
                        resid_corr=rc["mean_pairwise"], n_eval=s_ev["n"])
    print(f"wrote {tdir}/featstack_{dataset}_{split}{tag}.csv and _winners.npz")



def _rebuttal_dev_extras(dataset, preds, W, s_ev, ev_idx, xi, mi, st, xt, mt, sid, feats_tgt,
                         device, args, min_obs=100):
    """Rebuttal tables on the development eval half (test untouched).

    (1) every member's mse_per_var per clinical group (same >=100-observation floor as the paper),
    (2) combiner gain by measurement intensity: eval windows split into tertiles of the fraction of
        observed input cells, each tertile scored with the SAME fit-half weights (BenJ).
    """
    from harmonize import featstack as FS
    tdir = ROOT / "docs" / "tables"
    tag = ("" if args.split_seed == 42 else f"_seed{args.split_seed}") + (args.table_tag or "")
    names = [p.name for p in preds]
    n = s_ev["n"]; sse = s_ev["sse"]                      # (F,H), (F,H,M)
    tot_n = n.sum(axis=1)
    groups = FS.feature_groups(feats_tgt, n)
    rows = []
    for g, gm in groups.items():
        gm = np.asarray(gm) & (tot_n >= min_obs)
        if gm.sum() == 0:
            continue
        for j, nm in enumerate(names):
            pv = sse[:, :, j].sum(axis=1)[gm] / tot_n[gm]
            rows.append(dict(dataset=dataset, group=g, n_vars=int(gm.sum()), model=nm,
                             mse_per_var=round(float(pv.mean()), 5)))
    df = pd.DataFrame(rows)
    df["rank"] = df.groupby("group")["mse_per_var"].rank(method="min").astype(int)
    df.to_csv(tdir / f"featstack_{dataset}_val{tag}_models_by_group.csv", index=False)

    dens = np.asarray(mi[ev_idx]).reshape(len(ev_idx), -1).mean(axis=1)
    cuts = np.quantile(dens, [1 / 3, 2 / 3])
    strata = np.digitize(dens, cuts)
    keys = [k for k in ("best_single", "simplex_shrunk500_var", "rank_harmonic_top5", "mean",
                        "select_var") if k in W]
    irows = []
    for t in range(3):
        idx = ev_idx[strata == t]
        s_t = FS.accumulate_stats(preds, idx, xi, mi, st, xt, mt, device)
        sc = FS.score_weights({k: W[k] for k in keys}, s_t)
        # best single WITHIN the stratum, chosen on the fit-half ranking, i.e. W["best_single"];
        # also the per-stratum oracle single model, which is selected on the eval tertile itself
        per_m = (s_t["sse"].sum(axis=1) / np.clip(s_t["n"].sum(axis=1), 1, None)[:, None])
        keep = s_t["n"].sum(axis=1) > 0
        oracle = float(per_m[keep].mean(axis=0).min())
        base = sc["best_single"]["mse_per_var"]
        for k in keys:
            irows.append(dict(dataset=dataset, tertile=t + 1,
                              obs_frac_lo=round(float(dens[strata == t].min()), 4),
                              obs_frac_hi=round(float(dens[strata == t].max()), 4),
                              windows=int(len(idx)), stays=int(len(np.unique(sid[idx]))), combiner=k,
                              mse_per_var=round(sc[k]["mse_per_var"], 5),
                              delta_vs_best_single_pct=round(100 * (sc[k]["mse_per_var"] - base) / base, 2),
                              delta_vs_oracle_single_pct=round(100 * (sc[k]["mse_per_var"] - oracle) / oracle, 2)))
    idf = pd.DataFrame(irows)
    idf.to_csv(tdir / f"featstack_{dataset}_val{tag}_intensity.csv", index=False)
    print(idf.to_string(index=False))


def _final_dir():
    """docs/tables for the published FINAL run; docs/tables/final<tag>/ for a tagged one."""
    d = ROOT / "docs" / "tables"
    if TABLE_TAG:
        d = d / f"final{TABLE_TAG}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _per_stay_sse(pred, y, m, inv, S):
    """Per (stay, channel) sums of squared error, accumulated with np.add.at.

    The obvious implementation loops over stays with a boolean mask; that is O(S*N) and eICU has
    15 385 test stays against 211 251 windows. Scatter-add is O(N).
    """
    d2 = (np.where(m, pred - y, 0.0) ** 2).sum(axis=1)      # (N, F), summed over horizon
    sse = np.zeros((S, d2.shape[1]))
    np.add.at(sse, inv, d2)
    return sse


def _mse_per_var(sse, tot_n):
    ok = tot_n > 0
    return float((sse.sum(axis=0)[ok] / tot_n[ok]).mean())


def bootstrap_ci(sse_a, sse_b, n_stay, tot_n, n_boot=2000, seed=42, alpha=0.05):
    """Percentile CI for the RELATIVE difference, resampling stays with replacement.

    The counts move with the resample here (unlike the randomisation test, where the two systems
    share one mask and only the errors swap), so both numerator and denominator are recomputed.
    """
    rng = np.random.default_rng(seed)
    S = sse_a.shape[0]
    out = np.empty(n_boot)
    for i in range(n_boot):
        take = rng.integers(0, S, size=S)
        tn = n_stay[take].sum(axis=0)
        ok = tn > 0
        a = float((sse_a[take].sum(axis=0)[ok] / tn[ok]).mean())
        b = float((sse_b[take].sum(axis=0)[ok] / tn[ok]).mean())
        out[i] = 100.0 * (a - b) / b
    return float(np.quantile(out, alpha / 2)), float(np.quantile(out, 1 - alpha / 2))


def randomisation_test(sse_a, sse_b, tot_n, n_perm=10000, seed=42):
    """Paired approximate randomisation test (Noreen, 1989) over STAYS.

    H0: the two systems are exchangeable. Each stay independently keeps or swaps its two
    contributions and the FULL statistic is recomputed, which matters because mse_per_var is a
    ratio of sums per channel and then a mean over channels -- not a mean over units, so a
    per-stay-score approximation would be testing a different quantity. The observation counts are
    identical between systems (same mask), so only the error sums swap.
    """
    obs = _mse_per_var(sse_a, tot_n) - _mse_per_var(sse_b, tot_n)
    rng = np.random.default_rng(seed)
    S = sse_a.shape[0]
    hits = 0
    for _ in range(n_perm):
        flip = rng.random(S) < 0.5
        sa = np.where(flip[:, None], sse_b, sse_a).sum(axis=0)
        sb = (sse_a.sum(axis=0) + sse_b.sum(axis=0)) - sa
        ok = tot_n > 0
        d = float((sa[ok] / tot_n[ok]).mean()) - float((sb[ok] / tot_n[ok]).mean())
        if abs(d) >= abs(obs) - 1e-12:
            hits += 1
    return obs, (hits + 1) / (n_perm + 1)


def member_weights(dataset, W, stats, members, tdir=None, features=None):
    """How much weight does each POOL MEMBER carry, per combiner?

    `weight_concentration` answers "how spread is the weight" but never says WHOSE it is, so
    "which member does the ensemble actually rely on" could not be answered from the published
    tables. W[f, h, m] is already in hand at the call site; this writes the per-member view of
    it. Two numbers per (combiner, member):

      mean_weight   -- the member's weight averaged over observed channels. Sums to 1 across
                       members for a simplex combiner, so it reads directly as a share.
      share_top1_pct -- the fraction of channels where this member carries the LARGEST weight.
                       A member can matter a lot on a few channels and little on average, and
                       the two columns separate those cases.

    Unconstrained combiners (ridge_var*) carry signed weights that do not normalise, so their
    mean_weight is reported as-is and is not a share -- the same caveat as the NaN rows in
    weight_concentration.
    """
    n = stats["n"]
    live = n.sum(axis=1) > 0
    names = list(members)
    rows = []
    for cname, w in sorted(W.items()):
        wf = w[:, 0, :][live]                                  # hour-tied: hour 0 represents all
        wsum = wf.sum(axis=1, keepdims=True)
        simplexish = (wf >= -1e-9).all() and np.all(np.abs(wsum) > 1e-6)
        wn = wf / np.clip(wsum, 1e-300, None) if simplexish else wf
        top = np.argmax(wf, axis=1)
        for m, mname in enumerate(names):
            rows.append({"dataset": dataset, "combiner": cname, "member": mname,
                         "mean_weight": round(float(wn[:, m].mean()), 5),
                         "share_top1_pct": round(100.0 * float((top == m).mean()), 1),
                         "n_channels": int(live.sum()),
                         "is_share": bool(simplexish)})
    df = pd.DataFrame(rows)
    tdir = tdir or (ROOT / "docs" / "tables")
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f"member_weights_{dataset}{TABLE_TAG}.csv"
    df.to_csv(out, index=False)
    # Print only the headline combiner: the full table is 11 combiners x M members.
    head = "simplex_shrunk500_var"
    if head not in W:
        head = sorted(W)[0]
    sub = (df[df.combiner == head].sort_values("mean_weight", ascending=False)
           [["member", "mean_weight", "share_top1_pct"]])
    print(f"\n=== member weights {dataset}, combiner {head} "
          f"(mean_weight sums to 1 over members) ===")
    print(sub.to_string(index=False))
    return df


def weight_concentration(dataset, W, stats, tdir=None):
    """How concentrated is each combiner's weight vector, per channel?

    The claim "BMA collapses onto one member as the database grows" is a statement about the
    WEIGHTS, so measure the weights instead of arguing from five gain numbers. Reported as the mean
    largest weight and the mean normalised entropy over channels that carry any observation;
    entropy 1.0 = uniform over all M members, 0.0 = a single member, which is hard selection.
    """
    n = stats["n"]
    live = n.sum(axis=1) > 0
    rows = []
    for name, w in sorted(W.items()):
        wf = w[:, 0, :][live]                                  # hour-tied: hour 0 represents all
        # Entropy is only defined for a distribution. Unconstrained combiners (ridge_var*) carry
        # negative weights whose sum can be ~0, so dividing by it produced max_weight ~1e12 and
        # normalised entropies above 1 in the published CSV. Report NaN for those rather than a
        # number that looks like a measurement: a signed weight vector has no entropy.
        wsum = wf.sum(axis=1, keepdims=True)
        simplexish = (wf >= -1e-9).all() and np.all(np.abs(wsum) > 1e-6)
        if not simplexish:
            rows.append({"dataset": dataset, "combiner": name, "n_channels": int(live.sum()),
                         "max_weight": float("nan"), "entropy_norm": float("nan"),
                         "eff_members": float("nan")})
            continue
        wf = wf / np.clip(wsum, 1e-300, None)
        M = wf.shape[1]
        ent = -(wf * np.log(np.clip(wf, 1e-300, None))).sum(axis=1) / np.log(M)
        rows.append({"dataset": dataset, "combiner": name, "n_channels": int(live.sum()),
                     "max_weight": round(float(wf.max(axis=1).mean()), 4),
                     "entropy_norm": round(float(ent.mean()), 4),
                     "eff_members": round(float(np.exp(ent * np.log(M)).mean()), 2)})
    df = pd.DataFrame(rows)
    tdir = tdir or (ROOT / "docs" / "tables")
    tdir.mkdir(parents=True, exist_ok=True)
    out = tdir / f"combiner_concentration_{dataset}{TABLE_TAG}.csv"
    df.to_csv(out, index=False)
    print(f"\n=== weight concentration {dataset} (entropy 1.0 = uniform, 0.0 = hard selection) ===")
    print(df.to_string(index=False))
    print(f"wrote {out}")
    return df


def _patient_of_stay(dataset, stays):
    """Map test stay ids to patient ids, so resampling can be done per PATIENT.

    Splits are by patient, so there is no leakage -- but 29-37 % of test stays in MIMIC-III,
    MIMIC-IV and eICU come from patients with several stays, and stays of one patient are not
    independent. Resampling stays as if they were understates the variance (pmk4). HiRID cannot be
    linked (stay == patient), so its patient level equals its stay level.
    """
    from pathlib import Path
    f = Path("data/long") / f"{dataset}_cohort.parquet"
    if not f.exists():
        log.warning("no cohort table for %s -- patient level falls back to stays", dataset)
        return np.asarray(stays)
    c = pd.read_parquet(f)
    if "stay_id" not in c.columns or "patient_id" not in c.columns:
        return np.asarray(stays)                       # HiRID: one stay per patient
    m = c[["stay_id", "patient_id"]].drop_duplicates().set_index("stay_id")["patient_id"]
    p = m.reindex(np.asarray(stays))
    if p.isna().any():
        raise SystemExit(f"{int(p.isna().sum())} test stays of {dataset} have no patient id")
    return p.to_numpy()


def _to_groups(arr, grp_inv, G):
    """Sum per-stay rows into per-group rows. The statistics are additive, so this is exact."""
    out = np.zeros((G,) + arr.shape[1:])
    np.add.at(out, grp_inv, arr)
    return out


def random_weight_baseline(dataset, stats_ev, best_i, n_draws=1000, seed=0):
    """FmGp: what does a RANDOMLY weighted ensemble of the same pool achieve?

    Closed form, no prediction pass: for an hour-tied weight vector w_f the masked SSE of channel f
    is sum_h (w'G w - 2 w'b + yy), from the evaluation-set statistics. Two variants: an independent
    Dirichlet(1) draw per channel, and one Dirichlet(1) vector shared by all channels.
    """
    G = stats_ev["G"].sum(axis=1)            # (F, M, M)  hour-tied
    b = stats_ev["b"].sum(axis=1)            # (F, M)
    yy = stats_ev["yy"].sum(axis=1)          # (F,)
    n = stats_ev["n"].sum(axis=1)            # (F,)
    ok = n > 0
    F, M = b.shape

    def mse_per_var(w):                      # w: (F, M)
        sse = np.einsum("fm,fmk,fk->f", w, G, w) - 2 * np.einsum("fm,fm->f", w, b) + yy
        return float((sse[ok] / n[ok]).mean())

    e = np.zeros((F, M)); e[:, best_i] = 1.0
    base = mse_per_var(e)
    rng = np.random.default_rng(seed)
    out = {}
    for kind in ("per_channel", "shared"):
        d = []
        for _ in range(n_draws):
            w = rng.dirichlet(np.ones(M), size=F if kind == "per_channel" else 1)
            if kind == "shared":
                w = np.repeat(w, F, axis=0)
            d.append(100.0 * (mse_per_var(w) - base) / base)
        d = np.asarray(d)
        out[kind] = dict(dataset=dataset, variant=kind, n_draws=n_draws,
                         mean_delta_pct=round(float(d.mean()), 2),
                         p05=round(float(np.percentile(d, 5)), 2),
                         p95=round(float(np.percentile(d, 95)), 2),
                         share_beating_best=round(float((d < 0).mean()), 3),
                         uniform_delta_pct=round(100.0 * (mse_per_var(np.full((F, M), 1.0 / M)) - base) / base, 2))
    return list(out.values())


def final_extras(dataset, preds, ev_preds, W, diag, xt, mt, sid, stats_ev, tdir,
                 feature_names, n_perm=10000, seed=42, block=20000):
    """Appendix artefacts for the FINAL (val-fit -> test-score) run.

    Weights are already fitted on validation; nothing here is selected on test. Predictions are
    streamed in row blocks because stacking all 14 members of the eICU test split would need
    ~26 GB.
    """
    from harmonize import featstack as FS      # imported lazily, as in _run_featstack
    N, H, F = xt.shape[0], xt.shape[1], xt.shape[2]
    M = len(ev_preds)
    names = [q.name for q in ev_preds]
    best_i = int(diag["best_m"])
    stays, inv = np.unique(sid, return_inverse=True)
    S = len(stays)

    # simplex_global and the matched equal-weight controls are here because the paired contrasts
    # below need their per-stay errors; select_var is here because the body quotes p-values for it.
    # The rank_harmonic_top<k> set is pool-size dependent: featstack derives k from the pool
    # (sorted({3, 5, 14, M})), so a 15-member pool has rank_harmonic_top15 and a 17-member one
    # top17. Naming only top14 here silently omitted the FULL-POOL harmonic from the CI and
    # randomisation tables for every extended pool -- the numbers existed in featstack_FINAL_*.csv
    # with no uncertainty beside them. Pick up every rank_harmonic_top* key that W actually has.
    _fixed = ("simplex_shrunk500_var", "simplex_var", "simplex_global", "greedy_var",
              "rank_harmonic_top5", "mean_top5_var", "rank_harmonic_top3",
              "mean_top3_var", "mean_top3", "mean", "select_var")
    _harm = sorted((k for k in W if k.startswith("rank_harmonic_top")),
                   key=lambda s: int(s.rsplit("top", 1)[1]))
    combos = [k for k in (*_fixed, *_harm) if k in W]
    # W is (F, H, M). Taking [:, 0, :] is only valid for hour-TIED weight sets; an hour-resolved
    # one would silently get its h=1 weights applied at all 24 steps, corrupting exactly the
    # per-horizon table meant to test whether the hour axis matters.
    for k in combos:
        assert np.allclose(W[k], W[k][:, :1, :]), f"{k} is hour-resolved; final_extras assumes tied"
    Ws = {k: W[k][:, 0, :] for k in combos}

    sse_stay = {k: np.zeros((S, F)) for k in combos}
    sse_stay["__best__"] = np.zeros((S, F))
    n_stay = np.zeros((S, F))
    sse_fh = np.zeros((F, H, M))          # per (variable, hour, model), for win shares
    n_fh = np.zeros((F, H))
    sse_h = {k: np.zeros((H, F)) for k in combos}
    sse_h["__best__"] = np.zeros((H, F))
    n_h = np.zeros((H, F))

    for s0 in range(0, N, block):
        s1 = min(s0 + block, N)
        # FS.accumulate_stats nan_to_nums each member; without the same guard one NaN member
        # turns the whole appendix NaN while the main FINAL table stays finite.
        Pb = np.nan_to_num(
            np.stack([np.asarray(q.obj)[s0:s1] for q in ev_preds], axis=-1).astype(np.float64))
        yb = np.nan_to_num(xt[s0:s1]); mb = mt[s0:s1]
        n_fh += mb.sum(axis=0).T
        n_h += mb.sum(axis=0)
        np.add.at(n_stay, inv[s0:s1], mb.sum(axis=1))
        d2 = np.where(mb[..., None], Pb - yb[..., None], 0.0) ** 2
        sse_fh += d2.sum(axis=0).transpose(1, 0, 2)
        for k in combos:
            e = np.einsum("nhfm,fm->nhf", Pb, Ws[k])
            sse_stay[k] += _per_stay_sse(e, yb, mb, inv[s0:s1], S)
            sse_h[k] += (np.where(mb, e - yb, 0.0) ** 2).sum(axis=0)
        eb = Pb[:, :, :, best_i]
        sse_stay["__best__"] += _per_stay_sse(eb, yb, mb, inv[s0:s1], S)
        sse_h["__best__"] += (np.where(mb, eb - yb, 0.0) ** 2).sum(axis=0)
        del Pb, d2

    tot_n = n_fh.sum(axis=1)
    base = _mse_per_var(sse_stay["__best__"], tot_n)

    # ---- uncertainty ------------------------------------------------------------------------
    rows = []
    for k in combos:
        obs, pval = randomisation_test(sse_stay[k], sse_stay["__best__"], tot_n,
                                       n_perm=n_perm, seed=seed)
        lo, hi = bootstrap_ci(sse_stay[k], sse_stay["__best__"], n_stay, tot_n, seed=seed)
        rows.append(dict(dataset=dataset, combiner=k,
                         mse_per_var=round(_mse_per_var(sse_stay[k], tot_n), 5),
                         best_single=round(base, 5), best_single_name=names[best_i],
                         delta_pct=round(100.0 * obs / base, 2),
                         ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                         p_value=round(pval, 5), n_perm=n_perm, n_stays=int(S)))
    pd.DataFrame(rows).to_csv(tdir / f"final_uncertainty_{dataset}.csv", index=False)

    # ---- PAIRED contrasts between combiners --------------------------------------------------
    # An interval against best-single says each combiner beats a single model. It does NOT say the
    # combiners differ from EACH OTHER, which is what the per-variable-vs-global and the
    # equal-vs-harmonic claims assert. Same stay draws, same eligible channels, so the pairing is
    # exact rather than a comparison of two separate intervals.
    PAIRS = [("simplex_var", "simplex_global"),          # does weighting per variable pay?
             ("simplex_shrunk500_var", "simplex_global"),  # ... for the estimator we report
             ("rank_harmonic_top5", "mean_top5_var"),    # does the decay profile pay?
             ("rank_harmonic_top3", "mean_top3_var"),
             ("simplex_shrunk500_var", "rank_harmonic_top5"),  # what does fitting buy over ranking?
             ("simplex_shrunk500_var", "mean_top5_var"),
             # pmk4: harm-14 reaches ~91 % of the ensemble's gain; is the remaining 9 % real?
             ("simplex_shrunk500_var", "rank_harmonic_top14"),
             ("rank_harmonic_top5", "rank_harmonic_top14")]
    prows = []
    for a, b in PAIRS:
        if a not in sse_stay or b not in sse_stay:
            print(f"  [paired: skipping {a} vs {b}, not scored]")
            continue
        obs, pval = randomisation_test(sse_stay[a], sse_stay[b], tot_n, n_perm=n_perm, seed=seed)
        lo, hi = bootstrap_ci(sse_stay[a], sse_stay[b], n_stay, tot_n, seed=seed)
        mb = _mse_per_var(sse_stay[b], tot_n)
        prows.append(dict(dataset=dataset, a=a, b=b,
                          mse_a=round(_mse_per_var(sse_stay[a], tot_n), 5), mse_b=round(mb, 5),
                          delta_pct=round(100.0 * obs / mb, 2),
                          ci_lo=round(lo, 2), ci_hi=round(hi, 2),
                          p_value=round(pval, 5), n_perm=n_perm, n_stays=int(S)))
    if prows:
        pd.DataFrame(prows).to_csv(tdir / f"final_paired_{dataset}.csv", index=False)
        print(f"\n=== paired contrasts {dataset} (a vs b, negative = a better) ===")
        print(pd.DataFrame(prows)[["a", "b", "delta_pct", "ci_lo", "ci_hi", "p_value"]]
              .to_string(index=False))

    # ---- PATIENT-level resampling (pmk4) -----------------------------------------------------
    # Same statistics, rows summed per patient: a cluster bootstrap / cluster randomisation test.
    # The stay-level tables above are kept unchanged so the published numbers stay reproducible.
    pat = _patient_of_stay(dataset, stays)
    pats, pinv = np.unique(pat, return_inverse=True)
    P = len(pats)
    sse_pat = {k: _to_groups(v, pinv, P) for k, v in sse_stay.items()}
    n_pat = _to_groups(n_stay, pinv, P)
    print(f"\n=== patient-level resampling {dataset}: {S} stays -> {P} patients ===")
    urows = []
    for k in combos:
        obs, pval = randomisation_test(sse_pat[k], sse_pat["__best__"], tot_n, n_perm=n_perm, seed=seed)
        lo, hi = bootstrap_ci(sse_pat[k], sse_pat["__best__"], n_pat, tot_n, seed=seed)
        urows.append(dict(dataset=dataset, combiner=k, unit="patient",
                          delta_pct=round(100.0 * obs / base, 2), ci_lo=round(lo, 2),
                          ci_hi=round(hi, 2), p_value=round(pval, 5), n_units=int(P)))
    pd.DataFrame(urows).to_csv(tdir / f"final_uncertainty_patient_{dataset}.csv", index=False)
    prows_p = []
    for a, b in PAIRS:
        if a not in sse_pat or b not in sse_pat:
            continue
        obs, pval = randomisation_test(sse_pat[a], sse_pat[b], tot_n, n_perm=n_perm, seed=seed)
        lo, hi = bootstrap_ci(sse_pat[a], sse_pat[b], n_pat, tot_n, seed=seed)
        mb_ = _mse_per_var(sse_pat[b], tot_n)
        prows_p.append(dict(dataset=dataset, a=a, b=b, unit="patient",
                            delta_pct=round(100.0 * obs / mb_, 2), ci_lo=round(lo, 2),
                            ci_hi=round(hi, 2), p_value=round(pval, 5), n_units=int(P)))
    if prows_p:
        pd.DataFrame(prows_p).to_csv(tdir / f"final_paired_patient_{dataset}.csv", index=False)
        print(pd.DataFrame(prows_p)[["a", "b", "delta_pct", "ci_lo", "ci_hi", "p_value"]]
              .to_string(index=False))

    # ---- random-weight baseline (FmGp) -------------------------------------------------------
    rrows = random_weight_baseline(dataset, stats_ev, best_i)
    pd.DataFrame(rrows).to_csv(tdir / f"final_random_{dataset}.csv", index=False)
    print(pd.DataFrame(rrows).to_string(index=False))

    # ---- win shares -------------------------------------------------------------------------
    win_fh = sse_fh.argmin(axis=-1)                    # (F, H)
    sup = n_fh >= 200
    tot = int(sup.sum())
    cnt = np.bincount(win_fh[sup].ravel(), minlength=M)
    pd.DataFrame(dict(dataset=dataset, model=names, cells_won=cnt,
                      share_pct=np.round(100.0 * cnt / max(tot, 1), 2), supported_cells=tot)
                 ).sort_values("cells_won", ascending=False)\
      .to_csv(tdir / f"final_winshare_{dataset}.csv", index=False)

    # ---- per-model, per-system ---------------------------------------------------------------
    groups = FS.feature_groups(feature_names, n_fh)      # already contains "all_charted"
    charted = tot_n > 0
    recs = []
    # Group means need the SAME >=100-observation floor the development tables use, and that the
    # paper states. Without it one HiRID medication channel with twelve observed cells supplies
    # 41 % of that group's mean, which is the artefact the filter exists to remove.
    MIN_GROUP_OBS = 100
    supported_var = tot_n >= MIN_GROUP_OBS
    for gname, gmask in groups.items():
        gm = np.asarray(gmask) & charted & supported_var
        if gm.sum() == 0:
            continue
        # Win shares must be counted over SUPPORTED CELLS ONLY. Selecting rows by
        # `gm & sup.any(axis=1)` keeps every one of a feature's 24 hours as soon as ONE hour is
        # supported, while the denominator counts supported cells — so shares exceeded 100 %. Worse,
        # an unsupported cell has all-zero SSE and argmin silently awards it to model index 0.
        cell = gm[:, None] & sup                          # (F, H) supported cells in this group
        denom = max(int(cell.sum()), 1)
        for j, nm in enumerate(names):
            sj = sse_fh[:, :, j].sum(axis=1)
            recs.append(dict(dataset=dataset, system=gname, model=nm, n_vars=int(gm.sum()),
                             mse_per_var=round(float((sj[gm] / tot_n[gm]).mean()), 5),
                             winshare_pct=round(100.0 * int((win_fh[cell] == j).sum()) / denom, 2),
                             supported_cells=denom, min_obs=MIN_GROUP_OBS))
        for k in combos:
            recs.append(dict(dataset=dataset, system=gname, model=f">>{k}", n_vars=int(gm.sum()),
                             mse_per_var=round(float((sse_stay[k].sum(axis=0)[gm] / tot_n[gm]).mean()), 5),
                             winshare_pct=np.nan))
    pd.DataFrame(recs).to_csv(tdir / f"final_bymodel_{dataset}.csv", index=False)

    # ---- per-horizon -------------------------------------------------------------------------
    hrows = []
    for k in combos:
        for h in range(H):
            ok = n_h[h] > 0
            a = float((sse_h[k][h][ok] / n_h[h][ok]).mean())
            b = float((sse_h["__best__"][h][ok] / n_h[h][ok]).mean())
            hrows.append(dict(dataset=dataset, combiner=k, horizon=h + 1,
                              mse_per_var=round(a, 5), best_single=round(b, 5),
                              delta_pct=round(100.0 * (a - b) / b, 2)))
    pd.DataFrame(hrows).to_csv(tdir / f"final_byhorizon_{dataset}.csv", index=False)
    print(f"wrote final_{{uncertainty,winshare,bymodel,byhorizon}}_{dataset}.csv")


def final_clinical(dataset, ev_preds, W, diag, ds_ev, xi, xt, mt, st, sid, tdir,
                   n_boot=2000, seed=42, block=20000):
    """SOFA / SAPS-II computed from the FINAL test forecasts (rebuttal, JFiH: clinical relevance).

    Scores are computed on the denormalised forecast vs the LOCF ground truth, exactly as the
    per-model clinical evaluation does. The combiner weights are the val-fitted ones; nothing is
    selected here. Patient-level paired bootstrap on the per-window SOFA-max absolute error.
    """
    feats_tgt, feats_in, feats_static = ds_ev.features_target, ds_ev.features_in, ds_ev.features_static
    N = xt.shape[0]
    seed_idx = [feats_in.index(f) if f in feats_in else -1 for f in feats_tgt]
    xi_last = denormalise(xi[:, -1:, :], feats_in, NORM)[:, 0, :]
    gt_phys = denormalise(xt, feats_tgt, NORM)
    sd = np.full((N, len(feats_tgt)), np.nan, dtype=np.float32)
    for j, k in enumerate(seed_idx):
        if k >= 0:
            sd[:, j] = xi_last[:, k]
    gt_locf = locf(np.where(mt, gt_phys, np.nan).astype(np.float32), axis=1, seed=sd)
    # 2026-11-03 (37.109/37.111). Same correction eval_forecast applies: for
    # MEDICATION_ZERO_CHANNELS no record means "not administered", so LOCF would leave an
    # unmeasured pressor at its train mean and the cardiovascular SOFA component would read
    # stage >= 2 in ~91 % of scored cells. This module writes docs/tables/ensemble_<ds>.json,
    # which run_eval.sh's eval_*.json glob never matches -- so it can NEVER be reached by
    # RERUN=akifix and has to carry the fix itself, or corrected and uncorrected SOFA numbers
    # end up compared side by side.
    from harmonize.eval_forecast import _medzero
    if not getattr(args, "no_medication_zero", False):
        gt_locf = _medzero(gt_locf, mt, feats_tgt)
    del gt_phys

    names = [q.name for q in ev_preds]
    preds = {"best_single:" + names[int(diag["best_m"])]: ("member", int(diag["best_m"]))}
    for i, nm in enumerate(names):
        if "persistence" in nm:
            preds["persistence"] = ("member", i)
    for k in ("simplex_shrunk500_var", "rank_harmonic_top5", "mean"):
        if k in W:
            preds[k] = ("combo", W[k][:, 0, :])
    rows, err = [], {}
    for lab, (kind, w) in preds.items():
        if kind == "member":
            pz = np.nan_to_num(np.asarray(ev_preds[w].obj, dtype=np.float32))
        else:
            pz = np.zeros(xt.shape, np.float32)
            for s0 in range(0, N, block):
                Pb = np.nan_to_num(np.stack([np.asarray(q.obj)[s0:s0 + block] for q in ev_preds],
                                            axis=-1).astype(np.float32))
                pz[s0:s0 + block] = np.einsum("nhfm,fm->nhf", Pb, w.astype(np.float32))
        sdf, sofa_m, saps_m = clinical_scores(pz, gt_locf, feats_tgt, dataset, np.asarray(st),
                                              feats_static, sid)
        del pz
        err[lab] = (sdf["pred_sofa_max"] - sdf["gt_sofa_max"]).abs().to_numpy()
        for mk, v in {**sofa_m, **saps_m}.items():
            if np.isscalar(v):
                rows.append(dict(dataset=dataset, predictor=lab, metric=mk, value=float(v)))
        log.info("clinical %s %s: sofa_max_mae %.4f  sapsii_mae %s", dataset, lab,
                 sofa_m.get("sofa_max_mae", np.nan), saps_m.get("sapsii_mae", "n/a"))
    pd.DataFrame(rows).to_csv(tdir / f"final_clinical_{dataset}.csv", index=False)

    # patient-level paired bootstrap: SOFA-max MAE difference (combiner - best single)
    pat = _patient_of_stay(dataset, np.asarray(sid))
    pu, pinv = np.unique(pat, return_inverse=True)
    P = len(pu); rng = np.random.default_rng(seed)
    bkey = next(k for k in err if k.startswith("best_single"))
    brows = []
    for lab, e in err.items():
        if lab == bkey:
            continue
        ok = np.isfinite(e) & np.isfinite(err[bkey])
        sa = np.bincount(pinv[ok], e[ok], P); sb = np.bincount(pinv[ok], err[bkey][ok], P)
        cn = np.bincount(pinv[ok], minlength=P).astype(float)
        bs = []
        for _ in range(n_boot):
            ix = rng.integers(0, P, P)
            bs.append((sa[ix].sum() - sb[ix].sum()) / cn[ix].sum())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        brows.append(dict(dataset=dataset, predictor=lab, vs=bkey, windows=int(ok.sum()), patients=P,
                          sofa_max_mae=round(float(e[ok].mean()), 4),
                          best_sofa_max_mae=round(float(err[bkey][ok].mean()), 4),
                          diff=round(float(e[ok].mean() - err[bkey][ok].mean()), 4),
                          ci_lo=round(float(lo), 4), ci_hi=round(float(hi), 4)))
    pd.DataFrame(brows).to_csv(tdir / f"final_clinical_paired_{dataset}.csv", index=False)
    print(pd.DataFrame(brows).to_string(index=False))


def final_support(dataset, ev_preds, W, diag, xt, mt, sid, eval_h5, feature_names, val_n,
                  block=20000, thresholds=(1, 10, 25, 50, 100, 200, 500, 1000)):
    """Channel support and sparsity-threshold sweep on the FINAL test predictions (pmk4, 6g9V).

    Re-scores predictions that already exist with val-fitted weights; nothing is selected on test.
    (1) per channel: window-level observed cells (what the metric counts; windows overlap at a 4 h
        stride, so one measurement is counted up to six times), unique measured hours, stays and
        patients contributing.
    (2) sparse-vs-dense pattern at several support thresholds, with tertiles defined both on the
        test counts (as in the paper) and on validation counts (independent of the scored set), and
        the result on the channels each threshold EXCLUDES.
    """
    import h5py
    F, H = xt.shape[2], xt.shape[1]
    names = [q.name for q in ev_preds]
    best = int(diag["best_m"])
    keys = [k for k in ("simplex_shrunk500_var", "rank_harmonic_top5", "rank_harmonic_top14") if k in W]
    Ws = {k: W[k][:, 0, :] for k in keys}
    sse = {k: np.zeros(F) for k in keys + ["best_single"]}
    sse_m = np.zeros((F, len(ev_preds)))          # per member, for the paper's within-group reference
    n_f = np.zeros(F)
    for s0 in range(0, xt.shape[0], block):
        Pb = np.nan_to_num(np.stack([np.asarray(q.obj)[s0:s0 + block] for q in ev_preds],
                                    axis=-1).astype(np.float64))
        yb = np.nan_to_num(xt[s0:s0 + block]); mb = np.asarray(mt[s0:s0 + block]).astype(bool)
        n_f += mb.sum(axis=(0, 1))
        sse["best_single"] += np.where(mb, Pb[..., best] - yb, 0.0).__pow__(2).sum(axis=(0, 1))
        sse_m += (np.where(mb[..., None], Pb - yb[..., None], 0.0) ** 2).sum(axis=(0, 1))
        for k in keys:
            e = np.einsum("nhfm,fm->nhf", Pb, Ws[k])
            sse[k] += np.where(mb, e - yb, 0.0).__pow__(2).sum(axis=(0, 1))
    with h5py.File(eval_h5, "r") as f:
        t0 = f["t0_hour"][:].astype(np.int64)
        win_in = int(f.attrs["window_in"])
    sid = np.asarray(sid).astype(np.int64)
    pat = _patient_of_stay(dataset, np.unique(sid))
    pmap = dict(zip(np.unique(sid), pat))
    rows = []
    mt_b = np.asarray(mt).astype(bool)
    for j, fn_ in enumerate(feature_names):
        w, h = np.nonzero(mt_b[:, :, j])
        if len(w) == 0:
            rows.append(dict(dataset=dataset, variable=fn_, window_cells=0, unique_hours=0,
                             stays=0, patients=0)); continue
        key = sid[w] * 1_000_000 + (t0[w] + win_in + h)
        st_ = np.unique(sid[w])
        rows.append(dict(dataset=dataset, variable=fn_, window_cells=int(len(w)),
                         unique_hours=int(len(np.unique(key))), stays=int(len(st_)),
                         patients=int(len(np.unique([pmap[x] for x in st_])))))
    sup = pd.DataFrame(rows)
    for k in ["best_single"] + keys:
        sup[f"mse_{k}"] = np.where(n_f > 0, sse[k] / np.maximum(n_f, 1), np.nan).round(5)
    sup.to_csv(_final_dir() / f"final_channel_support_{dataset}.csv", index=False)

    per = {k: sse[k] / np.maximum(n_f, 1) for k in sse}
    charted = n_f > 0
    trows = []
    for thr in thresholds:
        keep = charted & (n_f >= thr)
        groups = {"retained": keep, "excluded": charted & ~keep}
        for src, cnt in (("test_counts", n_f), ("val_counts", np.asarray(val_n, dtype=float))):
            if keep.sum() >= 6:
                q = np.quantile(cnt[keep], [1 / 3, 2 / 3])
                groups[f"sparse_third[{src}]"] = keep & (cnt <= q[0])
                groups[f"mid_third[{src}]"] = keep & (cnt > q[0]) & (cnt <= q[1])
                groups[f"dense_third[{src}]"] = keep & (cnt > q[1])
        for g, gm in groups.items():
            if gm.sum() == 0:
                continue
            b = float(per["best_single"][gm].mean())
            # the paper's Section 5 reference: the best pool member WITHIN the group
            pm = (sse_m[gm] / np.maximum(n_f[gm], 1)[:, None]).mean(axis=0)
            gb = float(pm.min())
            for k in keys:
                a = float(per[k][gm].mean())
                trows.append(dict(dataset=dataset, min_obs=thr, group=g, n_vars=int(gm.sum()),
                                  combiner=k, mse_per_var=round(a, 5), best_single=round(b, 5),
                                  delta_pct=round(100 * (a - b) / b, 2),
                                  group_best_member=names[int(pm.argmin())], group_best=round(gb, 5),
                                  delta_vs_group_best_pct=(round(100 * (a - gb) / gb, 2) if gb > 0 else np.nan)))
    tdf = pd.DataFrame(trows)
    tdf.to_csv(_final_dir() / f"final_threshold_{dataset}.csv", index=False)
    print(tdf[tdf.combiner == "simplex_shrunk500_var"].to_string(index=False))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Multi-model ensemble + orthogonality eval")
    ap.add_argument("--checkpoints", nargs="*", default=[], type=Path)
    ap.add_argument("--test-h5", required=True, type=Path)
    ap.add_argument("--norm-stats", type=Path, default=None)
    ap.add_argument("--long-parquet", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ortho-subsample", type=int, default=4000,
                    help="windows used for the residual-correlation matrix")
    # --- per-(variable, hour) stacking mode (§13.6b; Bosch et al. AutoML 2025) ---
    ap.add_argument("--mode", choices=["ensemble", "featstack"], default="ensemble",
                    help="'ensemble' = the mean-ensemble + orthogonality eval (§13.5); "
                         "'featstack' = per-(variable,hour) stacking with a stay-grouped "
                         "fit/eval split of the test set")
    ap.add_argument("--pred-files", nargs="*", default=[],
                    help="extra precomputed base models as name=path.npy (FMs: moirai/tirex2/mira)")
    ap.add_argument("--split-seed", type=int, default=42,
                    help="seed for the stay-grouped fit/eval split. Every result so far used the "
                         "default, so the reported gains rest on ONE partition of patients; vary "
                         "this to separate a real effect from a lucky split")
    ap.add_argument("--fit-frac", type=float, default=0.5,
                    help="featstack: share of STAYS used to fit the combiners")
    ap.add_argument("--no-medication-zero", action="store_true", default=False,
                    help="leave not-given medications where LOCF put them (z=0 -> the channel "
                         "train MEAN, a phantom infusion). Reproduces tables written before "
                         "2026-11-03; see 37.109. Must match the eval_forecast setting of any "
                         "table these numbers are placed beside.")
    ap.add_argument("--ridge-lambda", type=float, default=1.0)
    ap.add_argument("--final-extras", action="store_true",
                    help="on the FINAL path, also write the uncertainty, win-share, per-group "
                         "and per-horizon artefacts used by the paper's appendix")
    ap.add_argument("--n-perm", type=int, default=10000,
                    help="iterations for the paired approximate randomisation test")
    ap.add_argument("--eval-h5", default=None, type=Path,
                    help="FINAL confirmation: fit combiners on --test-h5 (the val split) and "
                         "score them on this split (test). Use ONCE, after the combiner is chosen")
    ap.add_argument("--eval-pred-files", nargs="*", default=[],
                    help="cached predictions for --eval-h5; must list the SAME models in the "
                         "SAME order as --pred-files (weights are indexed by position)")
    ap.add_argument("--group-min-obs", type=int, default=100,
                    help="channels below this many observed cells are excluded from the "
                         "equal-weight group means (they dominate mse_per_var on tiny support)")
    ap.add_argument("--dump-combined", default=None, metavar="DIR",
                    help="materialise the combined forecast trajectory for --combined-weight-set "
                         "and exit (for feeding the outcome heads). Weights are fitted on the fit "
                         "half of the split passed in --test-h5, which must be val.")
    ap.add_argument("--combined-weights-from", default=None, metavar="NPZ",
                    help="reuse weights saved by an earlier --dump-combined instead of refitting. "
                         "Required for every split after the first, so the ensemble is one fixed "
                         "function across train/val/test. The member list must match exactly.")
    ap.add_argument("--combined-weight-set", default="simplex_shrunk500_var",
                    help="which fitted weight set to materialise (default: the published ENSEMBLE)")
    ap.add_argument("--rows", choices=["all", "first-day"], default="all",
                    help="restrict to admission windows (t0_hour == 0). Dumped files get a '_fd' "
                         "split tag. Use for anything feeding the outcome heads, which are "
                         "first-day-only.")
    ap.add_argument("--dump-preds", default=None, metavar="DIR",
                    help="write each base model's predictions to DIR and exit; re-use them via "
                         "--pred-files so later combiner experiments need no GPU and no checkpoints")
    ap.add_argument("--n-clusters", type=int, default=6,
                    help="Tier-4 patient clusters (input-only features); 1 disables the tier")
    ap.add_argument("--cluster-mode", choices=["meta", "physio", "error", "router"], default="meta",
                    help="meta = density/severity/statics summary; physio = observed per-channel "
                         "means only (no statics, no missingness proxies); error = ORACLE "
                         "clustering on the per-model error profile (uses targets, not "
                         "deployable — an upper bound on any input-based router); "
                         "router = the deployable two-stage version (cluster on fit-half errors, "
                         "assign eval-half windows from input alone)")
    ap.add_argument("--cluster-top-n", type=int, default=24,
                    help="physio mode: how many best-covered channels form the state vector")
    ap.add_argument("--final-support", action="store_true",
                    help="FINAL path: channel support table and sparsity-threshold sweep (rebuttal)")
    ap.add_argument("--rebuttal-extras", action="store_true",
                    help="dev path: per-model x clinical-group table and intensity tertiles")
    ap.add_argument("--table-tag", default="",
                    help="suffix for the development tables, so robustness pools (rebuttal) never "
                         "overwrite the published featstack_<ds>_val.csv")
    ap.add_argument("--final-clinical", action="store_true",
                    help="FINAL path: SOFA/SAPS-II from the test forecasts of best single, "
                         "persistence and the main combiners (rebuttal)")
    ap.add_argument("--allow-test-fit", action="store_true",
                    help="deliberately fit the combiners on test (spends the held-out set)")
    ap.add_argument("--combiner-sweep", action="store_true",
                    help="also fit BMA at several effective-sample-size discounts and Hedge at "
                         "several learning rates, so the untuned-baseline objection is answerable")
    ap.add_argument("--min-support", type=int, default=200,
                    help="featstack: min observed cells for a per-(var,hour) rule; "
                         "thinner cells fall back to the per-variable rule")
    args = ap.parse_args()

    global NORM, TABLE_TAG
    TABLE_TAG = args.table_tag or ""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = args.test_h5.stem.split("_seed")[0]
    norm_path = args.norm_stats or (ROOT / "data" / "splits" / f"{dataset}_norm_stats.parquet")
    NORM = pd.read_parquet(norm_path).set_index("variable")

    ds = ICUDataset(str(args.test_h5), preload=True)
    feats_tgt, feats_static = ds.features_target, ds.features_static
    xi, mi, st, xt, mt, sid, t0 = gather_tensors(ds, args.batch_size, args.limit)
    n = len(xi)
    log.info("Test windows: %d | models: %d | dataset: %s", n, len(args.checkpoints), dataset)

    # Ground truth (model-independent): LOCF on x_target seeded from last input step
    feats_in = ds.features_in
    seed_idx = [feats_in.index(f) if f in feats_in else -1 for f in feats_tgt]
    xi_phys = denormalise(xi, feats_in, NORM)
    gt_phys = denormalise(xt, feats_tgt, NORM)
    seed = np.full((n, len(feats_tgt)), np.nan, dtype=np.float32)
    for j, k in enumerate(seed_idx):
        if k >= 0:
            seed[:, j] = xi_phys[:, -1, k]
    gt_locf = locf(np.where(mt, gt_phys, np.nan).astype(np.float32), axis=1, seed=seed)
    # 2026-11-03 (37.109/37.111). Same correction eval_forecast applies: for
    # MEDICATION_ZERO_CHANNELS no record means "not administered", so LOCF would leave an
    # unmeasured pressor at its train mean and the cardiovascular SOFA component would read
    # stage >= 2 in ~91 % of scored cells. This module writes docs/tables/ensemble_<ds>.json,
    # which run_eval.sh's eval_*.json glob never matches -- so it can NEVER be reached by
    # RERUN=akifix and has to carry the fix itself, or corrected and uncorrected SOFA numbers
    # end up compared side by side.
    from harmonize.eval_forecast import _medzero
    if not getattr(args, "no_medication_zero", False):
        gt_locf = _medzero(gt_locf, mt, feats_tgt)

    if args.mode == "featstack":
        _run_featstack(args, ds, device, xi, mi, st, xt, mt, sid, feats_tgt, dataset)
        return

    # subsample indices for orthogonality
    rng = np.random.default_rng(42)
    sub = rng.choice(n, size=min(args.ortho_subsample, n), replace=False)
    sub_mask = mt[sub].astype(bool)

    ens_sum = np.zeros((n, xt.shape[1], len(feats_tgt)), dtype=np.float32)
    per_model = {}
    resid_sub = {}   # model_name → flattened residual vector on subsample observed positions

    for ck in args.checkpoints:
        model, cfg = load_model(ck, ds, device)
        name = cfg.get("run_name", ck.name)
        pred_z = predict(model, xi, mi, st, device, args.batch_size)
        ens_sum += pred_z

        fc = forecast_err(pred_z, xt, mt)
        _, sofa_m, saps_m = clinical_scores(pred_z, gt_locf, feats_tgt, dataset,
                                            np.asarray(st), feats_static, sid)
        per_model[name] = {"model": cfg.get("model"), "forecast": fc,
                           "sofa": sofa_m, "sapsii": saps_m}
        # residuals on the orthogonality subsample (z-score, observed positions)
        r = np.where(sub_mask, pred_z[sub] - np.nan_to_num(xt[sub]), np.nan)
        resid_sub[name] = r
        log.info("  %-32s mse_pv=%.4f  ΔSOFA2 sens=%.3f", name,
                 fc["mse_per_var"], sofa_m.get("dSOFA2_sensitivity", float("nan")))
        del model, pred_z

    K = len(args.checkpoints)

    # ----- Orthogonality: mean pairwise residual correlation (over observed positions) -----
    names = list(resid_sub.keys())
    flat = {nm: resid_sub[nm][sub_mask] for nm in names}   # 1-D aligned vectors
    corr = np.full((K, K), np.nan)
    for a in range(K):
        for b in range(K):
            va, vb = flat[names[a]], flat[names[b]]
            ok = np.isfinite(va) & np.isfinite(vb)
            if ok.sum() > 10:
                corr[a, b] = np.corrcoef(va[ok], vb[ok])[0, 1]
    off = corr[~np.eye(K, dtype=bool)]
    mean_rho = float(np.nanmean(off)) if K > 1 else float("nan")

    # ----- Ensemble (mean) scores -----
    ens_z = ens_sum / K
    fc_e = forecast_err(ens_z, xt, mt)
    sofa_df_e, sofa_e, saps_e = clinical_scores(ens_z, gt_locf, feats_tgt, dataset,
                                                np.asarray(st), feats_static, sid)

    best_single = min(per_model.values(), key=lambda d: d["forecast"]["mse_per_var"])
    summary = {
        "dataset": dataset, "n_windows": int(n), "n_models": K,
        "models": [{"name": nm, **{k: per_model[nm][k] for k in ("model", "forecast", "sofa")}}
                   for nm in names],
        "orthogonality": {
            "mean_pairwise_residual_corr": mean_rho,
            "ensemble_gain_potential": (1 - mean_rho) if mean_rho == mean_rho else None,
            "corr_matrix": {names[a]: {names[b]: (None if np.isnan(corr[a, b]) else round(float(corr[a, b]), 3))
                                       for b in range(K)} for a in range(K)},
        },
        "ensemble_mean": {"forecast": fc_e, "sofa": sofa_e, "sapsii": saps_e},
        "best_single":   {"name": [nm for nm in names if per_model[nm] is best_single][0]
                          if any(per_model[nm] is best_single for nm in names) else "",
                          "forecast": best_single["forecast"], "sofa": best_single["sofa"]},
    }

    # Optional Sepsis-3 on the ensemble
    if args.long_parquet and args.long_parquet.exists():
        from harmonize.scores.sepsis3 import susinf_per_stay, susinf_coincident
        long_df = pd.read_parquet(args.long_parquet, columns=["stay_id", "variable", "hour", "value"])
        su = susinf_per_stay(long_df).set_index("stay_id")
        sofa_df_e["stay_id"] = sid
        # Sepsis-3 needs the ΔSOFA rise within ±24 h of the infection onset, not merely a stay
        # that had an infection at some point (§37.38 D19).
        _onset = sofa_df_e["stay_id"].map(
            su["susinf_onset_h"].where(su["has_susinf"], np.nan)).to_numpy(dtype=float)
        has = pd.Series(susinf_coincident(_onset, t0, ds.window_in, ds.window_out),
                        index=sofa_df_e.index)
        share = float(has.mean())
        if share >= 0.005:
            gt_s = (has & (sofa_df_e["gt_delta_sofa"] >= 2)).values
            pr_s = (has & (sofa_df_e["pred_delta_sofa"] >= 2)).values
            tp = int((gt_s & pr_s).sum()); fn = int((gt_s & ~pr_s).sum())
            tn = int((~gt_s & ~pr_s).sum()); fp = int((~gt_s & pr_s).sum())
            summary["ensemble_mean"]["sepsis3"] = {
                "evaluable": True, "sensitivity": tp / max(tp + fn, 1),
                "specificity": tn / max(tn + fp, 1), "gt_prevalence": float(gt_s.mean())}
        else:
            summary["ensemble_mean"]["sepsis3"] = {"evaluable": False, "susinf_share": share}

    # ----- Report -----
    bs = summary["best_single"]
    log.info("=" * 64)
    log.info("Orthogonality: mean pairwise residual ρ = %.3f  (gain potential %.3f)",
             mean_rho, 1 - mean_rho if mean_rho == mean_rho else float("nan"))
    log.info("Best single (%s): mse_pv=%.4f  ΔSOFA2 sens=%.3f",
             bs["name"], bs["forecast"]["mse_per_var"], bs["sofa"].get("dSOFA2_sensitivity", float("nan")))
    log.info("Ensemble (mean): mse_pv=%.4f  ΔSOFA2 sens=%.3f",
             fc_e["mse_per_var"], sofa_e.get("dSOFA2_sensitivity", float("nan")))
    d_mse = 100 * (1 - fc_e["mse_per_var"] / bs["forecast"]["mse_per_var"])
    d_sens = sofa_e.get("dSOFA2_sensitivity", 0) - bs["sofa"].get("dSOFA2_sensitivity", 0)
    log.info("→ Ensemble vs best single: MSE %+.1f%%  |  ΔSOFA2 sensitivity %+.3f",
             d_mse, d_sens)
    log.info("=" * 64)

    summary["medication_zero"] = bool(not getattr(args, "no_medication_zero", False))
    out = args.out or (ROOT / "docs" / "tables" / f"ensemble_{dataset}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    log.info("Wrote ensemble summary → %s", out)


if __name__ == "__main__":
    main()
