"""
harmonize/models/lightgbm_fc.py
Gradient-boosted tree forecaster (LightGBM) as an additional pool member.

Added for the ML4H 2026 rebuttal: JFiH noted the pool has no tree ensembles and a single linear
model. This is a deliberately plain, non-neural member: one LightGBM regressor per target channel,
with the horizon step as a feature, on hand-built summaries of the 24 h input window. It writes
predictions in exactly the layout `ensemble_eval --dump-preds` produces -- (N, 24, 94) float32, in
z-scored space, rows in h5 order -- so it enters a combiner pool through `--pred-files`.

Protocol matches the other learned members: fit on train, early stopping on validation, test never
read during fitting. It is NOT part of the pre-registered 14-model pool; it is a robustness member
evaluated under the development protocol.

    python -m harmonize.models.lightgbm_fc --dataset umm
"""
from __future__ import annotations
import argparse, time
import json
from pathlib import Path
import h5py
import numpy as np

# Window summaries per channel. The input is forward-filled and z-scored, with 0 before the first
# observation; mask_in marks real observations, so every statistic is computed on observed cells.
N_STATS = 7


def window_features(x, m, static, static_mask):
    """x, m: (N, T, F) z-scored inputs and observation mask -> (N, F*7 + 2*S) features."""
    N, T, F = x.shape
    mf = m.astype(np.float32)
    cnt = mf.sum(1)                                            # (N, F)
    has = cnt > 0
    s = (x * mf).sum(1)
    mean = np.where(has, s / np.maximum(cnt, 1), np.nan)
    big = np.where(m, x, np.nan)
    with np.errstate(all="ignore"):
        mn = np.nanmin(big, axis=1); mx = np.nanmax(big, axis=1)
    last = x[:, -1, :]                                         # forward-filled = LOCF value
    last = np.where(has, last, np.nan)
    # hours since the last real observation (T if never observed in the window)
    idx = np.where(m, np.arange(T)[None, :, None], -1).max(1)
    since = np.where(idx >= 0, T - 1 - idx, T).astype(np.float32)
    # first observed value, for a crude within-window trend
    fidx = np.where(m, np.arange(T)[None, :, None], T).min(1)
    first = np.take_along_axis(x, np.clip(fidx, 0, T - 1)[:, None, :], 1)[:, 0, :]
    trend = np.where(cnt >= 2, last - first, np.nan)
    feats = [last, mean, mn, mx, cnt, since, trend]            # N_STATS of them
    st = np.where(static_mask, static, np.nan)
    return np.concatenate([f.astype(np.float32) for f in feats] + [st, static_mask.astype(np.float32)], 1)


def _load(h5, idx=None):
    with h5py.File(h5, "r") as f:
        sl = slice(None) if idx is None else np.sort(idx)
        return (f["x_in"][sl], f["mask_in"][sl], f["static"][sl], f["static_mask"][sl],
                f["x_target"][sl], f["mask_target"][sl])


def _rows(wf, xt, mt, f, rng, cap):
    """Expand window features to (window, horizon) rows where channel f is observed."""
    n, h = np.nonzero(mt[:, :, f])
    if cap and len(n) > cap:
        k = rng.choice(len(n), cap, replace=False); n, h = n[k], h[k]
    X = np.concatenate([wf[n], h[:, None].astype(np.float32)], 1)
    return X, xt[n, h, f].astype(np.float32)


def _labels(h5: str, head: str, idx=None):
    """The SAME label definition train_head uses, so the two backends are comparable.

    mortality : finite `mortality_hours` (survivors are NaN)       — always valid
    los3/los7 : `los_hours` > 72 / 168                              — NaN LoS is not evaluable
    """
    with h5py.File(h5, "r") as f:
        sl = slice(None) if idx is None else np.sort(idx)
        if head == "mortality":
            mh = f["mortality_hours"][sl]
            return (~np.isnan(mh)).astype(np.float32), np.ones(len(mh), bool)
        if head in ("los3", "los7"):
            lh = f["los_hours"][sl]
            thr = 72.0 if head == "los3" else 168.0
            return (lh > thr).astype(np.float32), np.isfinite(lh)
        if head == "aki":
            # D16: the same corrected KDIGO label train_head builds -- relative criteria PLUS the
            # absolute >=0.3 mg/dL 48h rise, which window_in + window_out = 24 + 24 h is exactly.
            # Gives section 16.10's dissociation a tree-head baseline on AKI as it now has on
            # dsofa2, so the hand-coded read-off is not measured against an MLP alone.
            from pathlib import Path as _P
            import pandas as _pd, h5py as _h5
            from harmonize.scores.kdigo import kdigo_stage
            with _h5.File(h5, "r") as g:
                fi = [x.decode() if isinstance(x, bytes) else x for x in g.attrs["features_in"]]
                ft = [x.decode() if isinstance(x, bytes) else x for x in g.attrs["features_target"]]
                ds_name = str(g.attrs["dataset"])
                ii, it = fi.index("creatinine"), ft.index("creatinine")
                xi = g["x_in"][sl][:, :, ii]; mi = g["mask_in"][sl][:, :, ii].astype(bool)
                xt = g["x_target"][sl][:, :, it]; mt = g["mask_target"][sl][:, :, it].astype(bool)
            # The H5 holds Z-SCORED values; kdigo_stage takes mg/dL, so denormalise first --
            # scoring the z-scores directly would compare a 0.3 mg/dL threshold against standard
            # deviations and silently stage almost everything.
            _ns = (_P(h5).resolve().parents[1] / "splits" / f"{ds_name}_norm_stats.parquet")
            _st = _pd.read_parquet(_ns).set_index("variable")
            _mu = float(_st.loc["creatinine", "train_mean"])
            _sd = float(_st.loc["creatinine", "train_std"])
            xi = xi * _sd + _mu
            xt = xt * _sd + _mu
            base = np.where(mi, xi, np.inf).min(1)
            base = np.where(mi.any(1), base, np.nan)
            worst = np.where(mt, xt, -np.inf).max(1)
            st = kdigo_stage(worst, base, creat_rise_48h=worst - base)
            ok = mt.any(1) & np.isfinite(base)
            return (np.nan_to_num(st, nan=0.0) >= 1).astype(np.float32), ok
        if head in ("dsofa2", "dsofa3"):
            # The SAME precomputed per-window label train_head and eval_forecast share, so the
            # tree head is directly comparable to §16.10's MLP head and to its read-off. NaN =
            # not computable, masked out rather than counted as a negative.
            lbl = f[head][sl]
            return np.nan_to_num(lbl).astype(np.float32), np.isfinite(lbl)
    raise ValueError(f"_labels: unsupported head {head!r}")


def cmd_classify(a) -> None:
    """LightGBM for DIRECT label prediction, and for read-off from a forecast trajectory.

    Why this exists. The §16.10/§16.10a dissociation is measured against ONE discriminative
    architecture: `_WindowEncoder`, a flat MLP over the flattened window with no temporal
    structure. If a gradient-boosted tree on the same window beats it, the claim "forecasts beat
    direct classification" has to become "the MLP is weak", so the baseline needs a second model
    class before the dissociation can be trusted.

    Two modes, and together they make the comparison architecture-independent:
      --traj-from absent  : features from the INPUT window      (tree, direct classification)
      --traj-from <npy>   : features from a forecast TRAJECTORY (tree, learned read-off)
    The feature construction, the label and the split are identical in both, so the difference is
    only which tensor the features come from — exactly the contrast `raw` vs `readoff` draws for
    the MLP.
    """
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score
    ds, out = a.dataset, {}
    H5 = {s: f"data/dataset/{ds}_seed42_{s}__all_4h_fwd.h5" for s in ("train", "val", "test")}
    for split in ("train", "val", "test"):
        sel = None
        if a.rows == "first-day":
            with h5py.File(H5[split], "r") as f:
                sel = np.flatnonzero(f["t0_hour"][:] == 0)
        if a.traj_from:
            # one path per split, in train,val,test order — row-aligned to the first-day subset
            tp = a.traj_from.split(",")[{"train": 0, "val": 1, "test": 2}[split]]
            tr = np.load(tp, mmap_mode="r")
            with h5py.File(H5[split], "r") as f:
                st, sm = f["static"][:], f["static_mask"][:]
            if sel is not None:
                st, sm = st[np.sort(sel)], sm[np.sort(sel)]
            if len(tr) != len(st):
                raise SystemExit(f"{tp}: {len(tr)} rows != {len(st)} windows — dump it with "
                                 "--rows first-day so it matches the subset")
            # A forecast is dense, so passing an all-ones mask is semantically right for the
            # trajectory itself -- but it makes `count` and `since` CONSTANT, killing 188 of 666
            # features (measured: 405 informative against the window arm's 557). `since` is the
            # informative-missingness signal, i.e. plausibly the main reason trees do well here, so
            # a traj arm without it is handicapped by construction and the comparison would
            # conflate "forecast instead of window" with "no missingness features".
            #
            # --traj-plus-mask appends the observation-pattern features computed from the INPUT
            # window, which a deployed system would still have. The arms then differ only in the
            # VALUES (predicted future vs observed past), which is the question.
            X = window_features(np.asarray(tr, np.float32),
                                np.ones(tr.shape, bool), st, sm)
            if a.traj_plus_mask:
                xi2, mi2, _, _, _, _ = _load(H5[split], sel)
                Xm = window_features(xi2, mi2, st, sm)
                F = xi2.shape[2]
                keep = np.concatenate([Xm[:, 4 * F:5 * F], Xm[:, 5 * F:6 * F]], axis=1)  # count, since
                X = np.concatenate([X, keep.astype(np.float32)], axis=1)
        else:
            xi, mi, stt, smm, _, _ = _load(H5[split], sel)
            X = window_features(xi, mi, stt, smm)
        y, ok = _labels(H5[split], a.head, sel)
        out[split] = (X[ok], y[ok])
        print(f"[{ds}/{split}] {X[ok].shape[0]:,} rows x {X.shape[1]} features, "
              f"pos_rate={y[ok].mean():.3f}", flush=True)

    Xtr, ytr = out["train"]; Xva, yva = out["val"]; Xte, yte = out["test"]
    params = dict(objective="binary", metric="auc", num_leaves=a.num_leaves,
                  learning_rate=a.learning_rate, min_data_in_leaf=a.min_data_in_leaf,
                  feature_fraction=a.feature_fraction, bagging_fraction=a.bagging_fraction,
                  bagging_freq=1, lambda_l2=a.lambda_l2, num_threads=a.threads,
                  verbosity=-1, seed=a.seed)
    rounds = a.fixed_rounds or 500
    b = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=rounds,
                  valid_sets=[lgb.Dataset(Xva, label=yva)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    res = {"dataset": ds, "head": a.head, "n_features": int(Xtr.shape[1]),
           "best_iteration": int(b.best_iteration or rounds),
           "traj_from": a.traj_from or "(input window)", "rows": a.rows, "seed": a.seed}
    for nm, (X, y) in (("val", (Xva, yva)), ("test", (Xte, yte))):
        s = b.predict(X, num_iteration=b.best_iteration)
        res[f"{nm}_auroc"] = float(roc_auc_score(y, s))
        res[f"{nm}_auprc"] = float(average_precision_score(y, s))
        res[f"{nm}_n"] = int(len(y)); res[f"{nm}_pos_rate"] = float(y.mean())
    # name the arm by WHICH forecaster fed it, so a denseAR-fed tree cannot be confused with a
    # LightGBM-fed one -- the whole point of the matched comparison
    if a.traj_from:
        src = Path(a.traj_from.split(",")[0]).name
        tag = "traj" if src.startswith("lightgbmfix_traj") else "trajdensear"
        if a.traj_plus_mask: tag += "pm"
    else:
        tag = "window"
    dest = Path("docs/tables") / f"lgbmhead_{ds}_{a.head}_{tag}_s{a.seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res, indent=2))
    print(f"[{ds}] {a.head} {tag} seed {a.seed}: "
          f"val AUROC {res['val_auroc']:.4f}  test AUROC {res['test_auroc']:.4f}  -> {dest}")


def main():
    import lightgbm as lgb
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out-dir", default="data/base_preds")
    ap.add_argument("--max-train-windows", type=int, default=150_000,
                    help="subsample of training windows (eICU has 685k); features are per window")
    ap.add_argument("--max-rows", type=int, default=400_000, help="cap on rows per channel")
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--es-holdout", type=float, default=0.0,
                    help="early-stop on this fraction of TRAINING stays (stay-grouped) instead of "
                         "val, so val is never seen during fitting (rebuttal leakage control)")
    # Hyper-parameters. Hardcoded until 2026-10-13; exposed so they can be swept. The defaults are
    # the untuned set every published LightGBM number used -- do not change them, pass overrides.
    ap.add_argument("--num-leaves", type=int, default=63)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--min-data-in-leaf", type=int, default=50)
    ap.add_argument("--feature-fraction", type=float, default=0.5)
    ap.add_argument("--bagging-fraction", type=float, default=0.8)
    ap.add_argument("--lambda-l2", type=float, default=1.0)
    ap.add_argument("--classify", action="store_true",
                    help="train a LightGBM CLASSIFIER for --head instead of the forecaster: the "
                         "second model class the §16.10 dissociation needs (see cmd_classify)")
    ap.add_argument("--head", default="mortality",
                    choices=["mortality", "los3", "los7", "dsofa2", "dsofa3", "aki"])
    ap.add_argument("--traj-plus-mask", action="store_true",
                    help="--traj-from: also append the INPUT window's count/since features, so the "
                         "traj arm is not handicapped by its dense mask (see cmd_classify)")
    ap.add_argument("--traj-from", default=None,
                    help="--classify: build features from these forecast trajectories "
                         "(train,val,test .npy paths) instead of the input window")
    ap.add_argument("--splits", nargs="+", default=["val", "test"],
                    help="which splits to predict. Add 'train' for the outcome-head arms "
                         "(§16.10a), which need a trajectory for the head's training rows. Note "
                         "that train-split predictions are IN-SAMPLE for this model.")
    ap.add_argument("--rows", choices=["all", "first-day"], default="all",
                    help="restrict the predicted rows to admission windows (t0_hour == 0), in "
                         "ascending dataset order, and tag the output '_fd'. Matches "
                         "ensemble_eval --rows and train_head --first-day-only.")
    ap.add_argument("--tag-suffix", default="", help="appended to the output tag so sweep "
                    "configurations do not overwrite each other")
    ap.add_argument("--fixed-rounds", type=int, default=0,
                    help="no early stopping at all: the same number of trees for every channel, "
                         "chosen a priori; val is never read during fitting (rebuttal control)")
    a = ap.parse_args()
    if a.classify:
        cmd_classify(a); return
    ds, rng = a.dataset, np.random.default_rng(a.seed)
    _sfx = a.tag_suffix
    tag = (f"lightgbmfix{_sfx}_{ds}" if a.fixed_rounds > 0 else
           f"lightgbmtr{_sfx}_{ds}" if a.es_holdout > 0 else f"lightgbm{_sfx}_{ds}")
    H5 = {s: f"data/dataset/{ds}_seed42_{s}__all_4h_fwd.h5" for s in ("train", "val", "test")}

    with h5py.File(H5["train"], "r") as f:
        ntr = f["x_in"].shape[0]
    tr_idx = rng.choice(ntr, min(ntr, a.max_train_windows), replace=False)
    t0 = time.time()
    xi, mi, st, sm, xt, mt = _load(H5["train"], tr_idx)
    wf_tr = window_features(xi, mi, st, sm); del xi, mi
    if a.es_holdout > 0:
        with h5py.File(H5["train"], "r") as f:
            sid = f["stay_id"][np.sort(tr_idx)]
        us = np.unique(sid)
        es_st = rng.choice(us, max(1, int(a.es_holdout * len(us))), replace=False)
        es = np.isin(sid, es_st)
        wf_va, xtv, mtv = wf_tr[es], xt[es], mt[es]
        wf_tr, xt, mt = wf_tr[~es], xt[~es], mt[~es]
        print(f"[{ds}] early stopping on {es.sum()} windows of {len(es_st)} held-out TRAIN stays; "
              f"val untouched", flush=True)
    elif a.fixed_rounds > 0:
        wf_va = None
        print(f"[{ds}] fixed {a.fixed_rounds} rounds per channel, no early stopping; val untouched",
              flush=True)
    else:
        xiv, miv, stv, smv, xtv, mtv = _load(H5["val"])
        wf_va = window_features(xiv, miv, stv, smv); del xiv, miv
    F = xt.shape[2]; T = xt.shape[1]
    print(f"[{ds}] features: train {wf_tr.shape} val {None if wf_va is None else wf_va.shape}  ({time.time()-t0:.0f}s)", flush=True)

    params = dict(objective="regression", learning_rate=a.learning_rate, num_leaves=a.num_leaves,
                  min_data_in_leaf=a.min_data_in_leaf, feature_fraction=a.feature_fraction,
                  bagging_fraction=a.bagging_fraction, bagging_freq=1, lambda_l2=a.lambda_l2,
                  num_threads=a.threads, verbose=-1, seed=a.seed)
    _shown = {k: v for k, v in params.items() if k not in ("num_threads", "verbose")}
    print(f"[{ds}] params: {_shown}", flush=True)
    boosters = {}
    for f in range(F):
        Xtr, ytr = _rows(wf_tr, xt, mt, f, rng, a.max_rows)
        if a.fixed_rounds > 0:
            if len(ytr) < 200:
                continue
            b = lgb.train(params, lgb.Dataset(Xtr, ytr), num_boost_round=a.fixed_rounds)
            b.best_iteration = a.fixed_rounds
            boosters[f] = b
            print(f"[{ds}] ch {f:2d}  rows {len(ytr):7d}  trees {a.fixed_rounds:4d}  (fixed)", flush=True)
            continue
        Xva, yva = _rows(wf_va, xtv, mtv, f, rng, a.max_rows)
        if len(ytr) < 200 or len(yva) < 50:
            continue                                   # never/barely charted: predict the mean (0)
        dtr = lgb.Dataset(Xtr, ytr); dva = lgb.Dataset(Xva, yva, reference=dtr)
        b = lgb.train(params, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        boosters[f] = b
        print(f"[{ds}] ch {f:2d}  rows {len(ytr):7d}  trees {b.best_iteration:4d}  "
              f"val mse {b.best_score['valid_0']['l2']:.4f}", flush=True)
    del wf_tr, xt, mt

    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        sel = None
        if a.rows == "first-day":
            with h5py.File(H5[split], "r") as f:
                sel = np.flatnonzero(f["t0_hour"][:] == 0)
            if sel.size == 0:
                raise SystemExit(f"--rows first-day selected 0 windows in {split}")
        xi, mi, st, sm, _, _ = _load(H5[split], sel)
        N = xi.shape[0]
        if sel is not None:
            print(f"[{ds}/{split}] --rows first-day: -> {N} windows", flush=True)
        pred = np.zeros((N, T, F), np.float32)
        for c0 in range(0, N, 20_000):
            wf = window_features(xi[c0:c0+20_000], mi[c0:c0+20_000], st[c0:c0+20_000], sm[c0:c0+20_000])
            for f, b in boosters.items():
                for h in range(T):
                    X = np.concatenate([wf, np.full((len(wf), 1), h, np.float32)], 1)
                    pred[c0:c0+len(wf), h, f] = b.predict(X, num_iteration=b.best_iteration)
        suffix = "_fd" if a.rows == "first-day" else ""
        dest = Path(a.out_dir) / f"{tag}__{ds}_{split}{suffix}.npy"
        np.save(dest, pred)
        print(f"[{ds}] wrote {dest}  {pred.shape}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
