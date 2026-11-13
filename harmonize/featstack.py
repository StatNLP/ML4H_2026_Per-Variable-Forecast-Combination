"""
harmonize/featstack.py — per-(variable, hour) stacking ensemble for the Paper-1 model ladder.

WHY THIS EXISTS, AND WHAT §13.6a GOT WRONG
------------------------------------------
§13.6a already asked "would a per-variable switching ensemble help?" and answered *mostly no*:
the per-dataset oracle gain (+12-24 %) turned out to be **entirely selection noise** (noise
fraction 1.0). That analysis picked the per-variable winner and scored it on **the same data**,
so it measured how well one can fit noise, not how well routing generalises.

This module fixes exactly that, following Bosch et al. (AutoML 2025, arXiv:2511.15350), whose
central methodological point is that stacker weights must be fitted on **out-of-fold** data.
Their Eq. 2 defines weights w_{i,h,q,m} over item / horizon / quantile / model with arbitrary
parameter tying; what we build here is the **w_{feature, hour, model}** tying — one weight per
(variable, forecast hour, base model).

The split is **stay-grouped**, never by window: windows from one ICU stay overlap heavily, so a
random window split puts near-duplicate rows on both sides and flatters the stacker. Grouping by
`stay_id` is what makes the held-out half honest.

WHERE THE WEIGHTS ARE ACTUALLY FITTED (corrected 2026-10-26, §37.43)
An earlier version of this header said the fit/eval split is taken on the **test** set, reasoning
that base models were selected on val so val is not out-of-fold w.r.t. model selection. That
design was abandoned and the header was never updated, so it described a protocol that fits
combiner weights on test — which this project forbids and which `ensemble_eval` actively refuses
(it rejects a combiner fit when the split is `test` unless `--allow-test-fit`, never used).

What every stored weight file and all 30 provenance sidecars actually record is
`fitted_on_split = train_fd`: the weights are fitted on the fit half of the **train, first-day**
split and then merely *applied* to train / val / test. `_fd` keeps one window per stay (the
admission window), so no two fit windows share a stay and the grouping is automatic. Neither val
nor test is ever used to fit or select a combiner, which is stricter than the superseded design
and stricter than the Bosch et al. protocol requires.

MEMORY: predictions are never materialised. For eICU (211 k windows x 24 h x 94 vars) one model
is ~1.9 GB and there are ~16 of them. Instead we stream chunks and accumulate per-(variable,
hour) **sufficient statistics** — Gram matrix G[f,h,m,m'], cross term b[f,h,m], per-model SSE —
which total ~0.6 M floats regardless of dataset size. Least squares per cell is then closed-form.

Everything is in z-score space and **masked**: only `mask_target` cells contribute, exactly as
the project-wide masked loss does (§13.1). Reported metric is `mse_per_var` (per-feature MSE
averaged over features), the project's selection metric.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


# ----------------------------------------------------------------------------- base predictors
class Predictor:
    """A base model that can score an arbitrary chunk of windows -> (n, H, F) z-scored."""

    def __init__(self, name: str, kind: str, obj):
        self.name, self.kind, self.obj = name, kind, obj

    @torch.no_grad()
    def predict(self, idx: np.ndarray, x_in, mask_in, static, device) -> np.ndarray:
        if self.kind == "npy":                     # precomputed FM, aligned to test row order
            return np.asarray(self.obj[idx], dtype=np.float32)
        out = self.obj(
            torch.from_numpy(x_in[idx]).to(device),
            torch.from_numpy(mask_in[idx]).to(device),
            torch.from_numpy(static[idx]).to(device),
        )
        return out.cpu().numpy().astype(np.float32)


def stay_grouped_split(stay_id: np.ndarray, frac_fit: float = 0.5, seed: int = 42):
    """Split window indices by STAY (never by window) -> (fit_idx, eval_idx).

    Windows from the same stay overlap in time and share a patient; splitting them at random
    would put near-duplicate rows on both sides and make the stacker look better than it is.
    """
    stays = np.unique(stay_id)
    rng = np.random.default_rng(seed)
    rng.shuffle(stays)
    n_fit = int(round(frac_fit * len(stays)))
    fit_stays = set(stays[:n_fit].tolist())
    is_fit = np.array([s in fit_stays for s in stay_id])
    return np.flatnonzero(is_fit), np.flatnonzero(~is_fit)


# ----------------------------------------------------------------- pass 1: sufficient statistics
def accumulate_stats(predictors, idx, x_in, mask_in, static, x_target, mask_target,
                     device, chunk: int = 2048):
    """Stream `idx` -> per-(feature, hour) Gram G, cross-term b, per-model SSE and counts."""
    M = len(predictors)
    H, F = x_target.shape[1], x_target.shape[2]
    G = np.zeros((F, H, M, M), dtype=np.float64)
    b = np.zeros((F, H, M), dtype=np.float64)
    sse = np.zeros((F, H, M), dtype=np.float64)
    n = np.zeros((F, H), dtype=np.float64)
    yy = np.zeros((F, H), dtype=np.float64)          # Sum y^2 -> lets us score any weight vector

    for s in range(0, len(idx), chunk):
        sub = idx[s:s + chunk]
        y = np.nan_to_num(x_target[sub]).astype(np.float32)     # (n,H,F)
        m = mask_target[sub].astype(np.float32)
        P = np.stack([p.predict(sub, x_in, mask_in, static, device) for p in predictors], axis=-1)
        P = np.nan_to_num(P) * m[..., None]                      # (n,H,F,M), unobserved zeroed
        ym = y * m
        # einsum over the window axis only -> per (H,F) blocks
        G += np.einsum("nhfi,nhfj->fhij", P, P, optimize=True)
        b += np.einsum("nhfi,nhf->fhi", P, ym, optimize=True)
        # P is already zeroed where unobserved, so (P - y*m)^2 contributes 0 on masked cells
        sse += ((P - ym[..., None]) ** 2).sum(axis=0).transpose(1, 0, 2)
        n += m.sum(axis=0).T
        yy += (ym ** 2).sum(axis=0).T
    return dict(G=G, b=b, sse=sse, n=n, yy=yy, names=[p.name for p in predictors])


# ------------------------------------------------------------------------------- weight solvers
def solve_ridge(G, b, lam: float = 1.0):
    """Per-cell ridge least squares w = (G + lam I)^-1 b. Returns (F,H,M)."""
    F, H, M, _ = G.shape
    W = np.zeros((F, H, M), dtype=np.float64)
    eye = np.eye(M)
    for f in range(F):
        for h in range(H):
            try:
                W[f, h] = np.linalg.solve(G[f, h] + lam * eye, b[f, h])
            except np.linalg.LinAlgError:
                W[f, h] = 0.0
    return W


def solve_simplex(G, b, iters: int = 300):
    """Per-variable weights constrained to the simplex (w >= 0, sum w = 1).

    This is the standard fix for the ridge blow-up: unconstrained least squares can hand a model
    a large negative weight and extrapolate, which is fine in-sample and catastrophic on new
    patients (MIMIC-III ridge: best mse_global, worst mse_per_var). Constraining to a convex
    combination means the stack can never leave the span of what the base models actually
    predict, so the worst case is bounded by the worst base model.

    Projected gradient on 0.5 w'Gw - w'b, which is exactly the SSE objective up to a constant.
    """
    F, M = b.shape
    W = np.full((F, M), 1.0 / M)
    for f in range(F):
        Gf, bf = G[f], b[f]
        L = float(np.linalg.eigvalsh(Gf).max()) or 1.0
        w = np.full(M, 1.0 / M)
        for _ in range(iters):
            w = _project_simplex(w - (Gf @ w - bf) / L)
        W[f] = w
    return W


def _project_simplex(v):
    """Euclidean projection onto {w >= 0, sum w = 1} (Duchi et al.)."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    theta = (css[rho] - 1) / (rho + 1.0)
    return np.maximum(v - theta, 0.0)


def greedy_selection(G, b, yy, n_iter: int = 25):
    """Caruana et al. greedy ensemble selection WITH REPLACEMENT, per variable.

    The default in several AutoML forecasting systems and a baseline in Bosch et al.: repeatedly
    add whichever base model most reduces the ensemble's error, allowing the same model to be
    picked again (which is how it expresses unequal weights). It only ever forms averages of
    base predictions, so like the harmonic blend it cannot extrapolate — but unlike the harmonic
    blend the weights are chosen by measured error reduction rather than by rank alone.
    """
    F, M = b.shape
    W = np.zeros((F, M))
    for f in range(F):
        Gf, bf, yf = G[f], b[f], yy[f]
        counts = np.zeros(M)
        for t in range(n_iter):
            best, best_sse = None, np.inf
            for m in range(M):
                c = counts.copy(); c[m] += 1
                w = c / c.sum()
                sse = float(w @ Gf @ w - 2.0 * w @ bf + yf)
                if sse < best_sse:
                    best_sse, best = sse, m
            counts[best] += 1
        W[f] = counts / counts.sum()
    return W


def pool_hours(stats):
    """Collapse the hour axis -> per-variable statistics (more support, less variance)."""
    return (stats["G"].sum(axis=1), stats["b"].sum(axis=1),
            stats["sse"].sum(axis=1), stats["n"].sum(axis=1))



# ------------------------------------------------------------------------- bagged (bootstrap) fits
def accumulate_stats_by_group(predictors, idx, x_in, mask_in, static, x_target, mask_target,
                              device, stay_id, n_groups: int = 50, seed: int = 42, chunk: int = 2048):
    """`accumulate_stats`, but kept as K disjoint partial sums over STAY groups.

    Every statistic here (G, b, sse, n, yy) is a plain sum over windows, so partial sums over a
    partition of the stays can be recombined by addition. That makes a bootstrap free: resample
    groups with replacement and add, instead of re-streaming the predictions B times. Grouping is by
    STAY (never by window) because windows from one stay overlap and would otherwise appear on both
    sides of a resample.
    """
    stays = np.unique(stay_id[idx])
    rng = np.random.default_rng(seed)
    rng.shuffle(stays)
    grp_of_stay = {s: i % n_groups for i, s in enumerate(stays.tolist())}
    gid = np.array([grp_of_stay[s] for s in stay_id[idx].tolist()])
    out = []
    for g in range(n_groups):
        sel = idx[gid == g]
        if len(sel) == 0:                      # more groups than stays
            continue
        out.append(accumulate_stats(predictors, sel, x_in, mask_in, static,
                                    x_target, mask_target, device, chunk))
    return out


def _pool(group_stats, take):
    keys = ("G", "b", "sse", "n", "yy")
    acc = {k: sum(group_stats[i][k] for i in take) for k in keys}
    acc["names"] = group_stats[0]["names"]
    return acc


def bagged_weight_sets(group_stats, B: int = 100, seed: int = 42):
    """Bootstrap-aggregated per-variable weights.

    Motivation: the per-channel fit is a high-variance estimator, and hard selection is the extreme
    case -- an argmin over M noisy estimates, where the "best model for this channel" is partly the
    luckiest model on the fitting sample. Averaging the weight TABLE over bootstrap resamples of the
    fitting stays is the standard variance reduction for exactly that, and for `select_var` it turns
    a brittle argmin into a soft vote (the share of resamples in which a model won the channel).

    Note this bags the WEIGHTS, not the predictions: the base models are untouched and the result is
    still a single (F,H,M) table applied exactly like any other combiner, so it costs nothing at
    inference.
    """
    K = len(group_stats)
    rng = np.random.default_rng(seed)
    F, H, M = group_stats[0]["b"].shape
    acc = {k: np.zeros((F, M)) for k in ("bagged_select_var", "bagged_simplex_var")}
    for _ in range(B):
        take = rng.integers(0, K, size=K)                  # groups, with replacement
        st = _pool(group_stats, take)
        per_var = st["sse"].sum(axis=1) / np.clip(st["n"].sum(axis=1), 1, None)[:, None]
        one = np.zeros((F, M))
        one[np.arange(F), per_var.argmin(axis=1)] = 1.0
        acc["bagged_select_var"] += one
        Gv = st["G"].sum(axis=1)                           # hour-tied
        bv = st["b"].sum(axis=1)
        acc["bagged_simplex_var"] += solve_simplex(Gv, bv)
    W = {}
    for k, v in acc.items():
        v = v / B
        v = v / np.clip(v.sum(axis=1, keepdims=True), 1e-12, None)
        W[k] = np.repeat(v[:, None, :], H, axis=1)
    return W



def rank_profile_weights(stats, ks=(3, 5, 14)):
    """Fit the SHAPE of the rank-weight profile instead of fixing it at 1, 1/2, 1/3, ...

    rank_harmonic_top-k assumes a specific decay. This asks what the optimal decay is: rank the
    members per channel as usual, then fit ONE weight vector over rank positions, shared by every
    channel and every database-specific ranking. That is k parameters in total rather than k per
    channel, so it has far less variance than a per-channel fit while being strictly more flexible
    than a fixed profile -- it can discover that the true decay is steeper, flatter, or not
    monotone.

    Cheap because the statistics are already per-model: permuting the model axis of G and b into
    rank order and summing over channels and hours gives the rank-space normal equations directly,
    with no pass over predictions.
    """
    G, b, sse, n = stats["G"], stats["b"], stats["sse"], stats["n"]
    F, H, M = b.shape
    per_var = sse.sum(axis=1) / np.clip(n.sum(axis=1), 1, None)[:, None]      # (F, M)
    order = np.argsort(per_var, axis=1)                                       # (F, M) rank -> model
    out, prof = {}, {}
    for k in ks:
        kk = min(k, M)
        sel = order[:, :kk]                                                   # (F, kk)
        # rank-space Gram and cross-term, summed over channels and hours
        Gr = np.zeros((kk, kk))
        br = np.zeros(kk)
        for f in range(F):
            if n[f].sum() <= 0:
                continue
            idx = sel[f]
            Gr += G[f][:, idx][:, :, idx].sum(axis=0)
            br += b[f][:, idx].sum(axis=0)
        w_rank = _project_simplex(np.linalg.solve(Gr + 1e-6 * np.eye(kk), br))
        W = np.zeros((F, M))
        np.put_along_axis(W, sel, np.broadcast_to(w_rank, (F, kk)), axis=1)
        out[f"rank_fitted_top{k}"] = np.repeat(W[:, None, :], H, axis=1)
        prof[f"top{k}"] = w_rank
    # Weight tensors and 1-D diagnostics are returned SEPARATELY: mixing them meant passing the
    # result straight to score_weights raised inside einsum on the profile entries.
    return out, prof


def bma_weights(per_var, n_f, n_eff=None, tau: float = 1.0):
    """Bayesian model averaging: posterior weights under a Gaussian likelihood, flat prior.

        w_m  proportional to  exp(-(N/2) * MSE_m / sigma^2),   sigma^2 estimated by the best member

    N is the effective sample size. Using the raw observed-cell count makes the posterior
    DEGENERATE here -- with 10^5-10^6 cells per channel the exponent is enormous and the posterior
    collapses onto a single model, i.e. BMA reduces to hard selection and inherits its weakness
    (Section 5.3, finding 4). Cells within a stay are anything but independent, so the honest
    effective sample size is the number of STAYS; pass it as `n_eff`. Both variants are reported,
    because the collapse is itself the result.

    `tau` divides the effective sample size. It exists because the ONE free choice in this
    estimator -- how much information a channel's cells really carry -- is not observable, and
    reporting an untuned posterior invites the objection that the collapse is an artefact of that
    choice rather than a property of BMA. tau=1 is the algorithm as specified; the sweep in
    Appendix C shows the collapse survives every setting that keeps the posterior a posterior.
    """
    F, M = per_var.shape
    sigma2 = np.clip(per_var.min(axis=1, keepdims=True), 1e-12, None)
    # n_eff may be a scalar (one stay count for the database) or, preferably, a per-channel
    # array: a channel observed in 12 stays must not be credited with the database's full
    # stay count, or its posterior collapses exactly as the raw cell count would make it.
    if n_eff is None:
        N = n_f.reshape(F, 1)
    else:
        _ne = np.asarray(n_eff, dtype=float)
        N = (np.full(F, float(_ne)) if _ne.ndim == 0 else _ne).reshape(F, 1)
    N = N / max(float(tau), 1e-12)
    z = -0.5 * N * (per_var / sigma2 - 1.0)          # subtract the best: shift-invariant, stable
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)


def hedge_weights(per_var, n_f, eta_scale: float = 1.0):
    """Exponentially weighted average (Hedge) with the standard learning rate.

    Hedge over T rounds with losses in [0,1] uses eta = sqrt(8 ln M / T) and guarantees regret
    <= sqrt(T ln M / 2) against the best fixed member. Here T is the observed-cell count of the
    channel and the per-cell loss is the squared error, normalised by the channel's mean MSE so the
    [0,1] scaling the bound assumes is approximately respected.

    The exponent then grows as sqrt(T) rather than T, so Hedge concentrates more slowly than BMA --
    it is the principled middle ground between uniform averaging and hard selection, which is
    exactly the axis this paper's ladder explores.

    `eta_scale` multiplies the learning rate. The bound assumes losses in [0,1]; normalising by the
    channel mean only approximates that, so the nominal eta is the right ORDER but not a derived
    value. Sweeping it is what separates "Hedge behaves this way" from "our normalisation does".
    """
    F, M = per_var.shape
    T = np.clip(n_f, 1.0, None).reshape(F, 1)
    scale = np.clip(per_var.mean(axis=1, keepdims=True), 1e-12, None)
    eta = float(eta_scale) * np.sqrt(8.0 * np.log(max(M, 2)) / T)
    z = -eta * T * (per_var - per_var.min(axis=1, keepdims=True)) / scale
    z -= z.max(axis=1, keepdims=True)
    w = np.exp(z)
    return w / np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)


def build_weight_sets(stats, lam: float, min_support: int, n_eff=None, sweep: bool = False):
    """All combiners as (F,H,M) weight tensors, so evaluation is one shared code path."""
    G, b, sse, n = stats["G"], stats["b"], stats["sse"], stats["n"]
    F, H, M = b.shape
    W = {}

    # per-model global score on the FIT half (per-variable MSE, the project metric)
    per_var = sse.sum(axis=1) / np.clip(n.sum(axis=1), 1, None)[:, None]   # (F,M)
    global_mse = per_var.mean(axis=0)                                      # (M,)
    best_m = int(np.argmin(global_mse))

    def const(vec):
        return np.broadcast_to(vec, (F, H, M)).copy()

    onehot = np.zeros(M); onehot[best_m] = 1.0
    W["best_single"] = const(onehot)
    W["mean"] = const(np.full(M, 1.0 / M))
    # GLOBAL top-k: members ranked pool-wide, not per channel. With the per-channel versions below
    # these form a 2x2 over (membership, weight profile), which is what separates "selecting the
    # right members per channel" from "weighting them unequally".
    for k in (3, 5):
        kk = min(k, M)
        gtop = np.argsort(global_mse)[:kk]
        v = np.zeros(M); v[gtop] = 1.0 / kk
        W[f"mean_top{k}"] = const(v)                       # global members, equal weights
        gh = 1.0 / np.arange(1, kk + 1); gh = gh / gh.sum()
        vh = np.zeros(M); vh[gtop] = gh
        W[f"rank_harmonic_gtop{k}"] = const(vh)            # global members, harmonic weights

    # inverse-MSE weights per variable (Bosch et al. h_inv), hour-tied
    inv = 1.0 / np.clip(per_var, 1e-8, None)
    inv = inv / inv.sum(axis=1, keepdims=True)
    W["inv_mse_var"] = np.repeat(inv[:, None, :], H, axis=1)

    # --- standard ensembling baselines the reviewer asked for --------------------------------
    nf = n.sum(axis=1)                                        # observed cells per channel
    W["bma_var"] = np.repeat(bma_weights(per_var, nf)[:, None, :], H, axis=1)
    if n_eff is not None:                                     # stay-level effective sample size
        W["bma_neff_var"] = np.repeat(bma_weights(per_var, nf, n_eff)[:, None, :], H, axis=1)
    W["hedge_var"] = np.repeat(hedge_weights(per_var, nf)[:, None, :], H, axis=1)
    if sweep:
        # Neither baseline was tuned while ENSEMBLE's kappa was chosen on validation. That
        # asymmetry is the fair-comparison objection, so sweep the one free knob of each over two
        # orders of magnitude. tau > 1 discounts the effective sample size (the within-stay
        # correlation the Gaussian likelihood ignores); eta_scale rescales Hedge's learning rate.
        for tau in (1e1, 1e2, 1e3, 1e4, 1e5):
            W[f"bma_tau{tau:g}_var"] = np.repeat(
                bma_weights(per_var, nf, tau=tau)[:, None, :], H, axis=1)
        for es in (1e-3, 1e-2, 1e-1, 1e1):
            W[f"hedge_eta{es:g}_var"] = np.repeat(
                hedge_weights(per_var, nf, eta_scale=es)[:, None, :], H, axis=1)

    # Fitted rank profiles. These were previously computed by a one-off script, so the numbers in
    # the appendix were not reproducible from the pipeline and existed only for the development
    # protocol. Registering them here puts them in both protocols and in the ladder.
    try:
        _rp, _ = rank_profile_weights(stats, ks=(3, 5, min(5, M) if M < 14 else 14))
        W.update(_rp)
    except Exception as _e:                 # a diagnostic must never take the ladder down
        print(f"  [rank profiles skipped: {_e}]")

    # GLOBAL simplex: a single weight vector shared by every channel, fitted on the same macro
    # objective as the per-variable version. Without it the paper cannot separate "combining helps"
    # from "combining PER VARIABLE helps" -- an ensemble can beat the best single model even when
    # one global vector would do. Minimising sum_f (w'G_f w - 2 w'b_f + c_f)/n_f over the simplex
    # is a simplex QP in the SUPPORT-NORMALISED statistics, so each channel contributes equally
    # rather than in proportion to how densely it is sampled.
    Gv_g, bv_g = G.sum(axis=1), b.sum(axis=1)                 # (F,M,M), (F,M) hour-pooled
    live = nf > 0
    inv = (1.0 / nf[live])[:, None, None]
    G_glob = (Gv_g[live] * inv).sum(axis=0)[None]             # (1,M,M)
    b_glob = (bv_g[live] * inv[:, :, 0]).sum(axis=0)[None]    # (1,M)
    w_glob = solve_simplex(G_glob, b_glob)[0]
    W["simplex_global"] = const(w_glob)

    # hard selection per variable / per (variable, hour)
    sel_v = np.zeros((F, M)); sel_v[np.arange(F), per_var.argmin(axis=1)] = 1.0
    W["select_var"] = np.repeat(sel_v[:, None, :], H, axis=1)
    cell_mse = sse / np.clip(n, 1, None)[..., None]
    sel_vh = np.zeros((F, H, M))
    fi, hi = np.meshgrid(np.arange(F), np.arange(H), indexing="ij")
    sel_vh[fi, hi, cell_mse.argmin(axis=2)] = 1.0
    # fall back to the per-variable choice where a cell is too thin to trust
    thin = n < min_support
    sel_vh[thin] = np.repeat(sel_v[:, None, :], H, axis=1)[thin]
    W["select_var_hour"] = sel_vh

    # RANK-HARMONIC top-k: rank the models per variable by fit-half MSE and blend the best k
    # with weights proportional to 1, 1/2, 1/3, ... 1/k. This is deliberately an *interpolation*
    # between the leaders rather than a hard pick: hard selection throws away the (small) chance
    # that the runner-up is right on a given window, and the per-variable winner is only modally
    # stable (~0.6 of hours), so committing fully to it commits to noise. Fixed weights also
    # cannot overfit — unlike ridge, nothing is estimated beyond the ranking itself.
    # k=14 was the whole pool when the pool had 14 members. It is NOT the whole pool in the
    # extended pools (15 or 17 members), where top-14 silently drops the lowest-ranked member(s)
    # per variable -- so M is added to the set to give a genuine full-pool harmonic blend
    # (harm-15, harm-17). Additive by construction: for M == 14 the set is unchanged at
    # {3, 5, 14}, so no previously published number moves.
    for k in sorted({3, 5, 14, M}):
        # clamp the COMPUTATION to the pool size but key on the requested k: keying on the clamped
        # value would make top3 and top5 the same entry for M<5 and silently drop one combiner
        kk = min(k, M)
        hw = 1.0 / np.arange(1, kk + 1)
        hw = hw / hw.sum()
        rank = np.argsort(per_var, axis=1)[:, :kk]          # (F,kk) best-kk model ids per variable
        Wk = np.zeros((F, M))
        np.put_along_axis(Wk, rank, np.broadcast_to(hw, (F, kk)), axis=1)
        W[f"rank_harmonic_top{k}"] = np.repeat(Wk[:, None, :], H, axis=1)
        # MATCHED equal-weight control: identical members, flat weights. The global `mean_top3`
        # above averages the three best members POOL-WIDE, so comparing it with harm-3 changes
        # membership and the weight profile at once and cannot isolate either.
        ew = np.full(kk, 1.0 / kk)
        Wke = np.zeros((F, M))
        np.put_along_axis(Wke, rank, np.broadcast_to(ew, (F, kk)), axis=1)
        W[f"mean_top{k}_var"] = np.repeat(Wke[:, None, :], H, axis=1)

    # learned linear stacks
    Gv, bv, _, nv = pool_hours(stats)
    Wv = np.zeros((F, M)); eye = np.eye(M)
    for f in range(F):
        try:
            Wv[f] = np.linalg.solve(Gv[f] + lam * eye, bv[f])
        except np.linalg.LinAlgError:
            Wv[f] = onehot
    W["ridge_var"] = np.repeat(Wv[:, None, :], H, axis=1)

    # --- Empirical-Bayes / James-Stein shrinkage toward a ranking prior -----------------------
    # Rationale (audit 2026-08-19): the per-variable fit has wildly uneven support — dense vitals
    # carry >100k observed cells while rare labs and episodic drugs carry <100. On the thin ones
    # the solver concentrates weight on whichever model happened to win a handful of points, which
    # is exactly the seed fragility measured on the small cohorts (UMM sd 4.18 pp, HiRID 2.48 pp).
    #
    #     w_shrunk[f] = a[f] * w_simplex[f] + (1 - a[f]) * w_prior[f],   a[f] = N_f / (N_f + N0)
    #
    # Both terms lie on the simplex, so the convex combination does too — the bounded-worst-case
    # property that makes simplex_var safe is preserved. This generalises the existing
    # `min_support` rule, which is the same idea as a hard step: dense variables keep the learned
    # weights, thin ones fall back, but now the transition is smooth and support-proportional
    # rather than a cliff at an arbitrary threshold.
    #
    # NB the harmonic prior is lower-variance, NOT zero-variance: its RANKING is estimated from the
    # same thin cells. The genuinely support-free fallback is the hour-pooled global weight vector,
    # which sees every variable's data at once, so that is what is used as N_f -> 0.
    Gv2, bv2, _, _ = pool_hours(stats)
    yv = stats["yy"].sum(axis=1)
    w_simplex = solve_simplex(Gv2, bv2)                       # (F, M)
    W["simplex_var"] = np.repeat(w_simplex[:, None, :], H, axis=1)

    n_f = stats["n"].sum(axis=1)                              # observed cells per variable (fit)
    w_prior = W["rank_harmonic_top5"][:, 0, :].copy()          # ranking prior
    pooled = solve_simplex(Gv2.sum(axis=0, keepdims=True), bv2.sum(axis=0, keepdims=True))[0]
    w_prior[n_f <= 0] = pooled                                 # no support at all -> global vector
    for N0 in (100, 500):
        alpha = (n_f / (n_f + float(N0)))[:, None]             # -> 1 dense, -> 0 thin
        w_sh = alpha * w_simplex + (1.0 - alpha) * w_prior
        W[f"simplex_shrunk{N0}_var"] = np.repeat(w_sh[:, None, :], H, axis=1)
    W["greedy_var"] = np.repeat(greedy_selection(Gv2, bv2, yv)[:, None, :], H, axis=1)

    Wvh = solve_ridge(G, b, lam)
    Wvh[thin] = np.repeat(Wv[:, None, :], H, axis=1)[thin]     # same thin-cell fallback
    W["ridge_var_hour"] = Wvh
    return W, dict(global_mse=global_mse, best_m=best_m, per_var=per_var, cell_mse=cell_mse)


# ---------------------------------------------------------------------------- pass 2: evaluation
def feature_groups(features, n, catalog_path=None):
    """Clinical groupings for reporting, so one number does not hide opposite effects.

    `mse_per_var` is a mean over features, so it is only meaningful once the denominator is the
    number of features the dataset ACTUALLY charts (channels it never records contribute 0.0 and
    would otherwise dilute the mean — the confound behind the ladder's optimistic per-dataset
    levels). Every group below is averaged over its OBSERVED members only.

    Groups: catalog `type` (lab / vital / medication / ventilation / derived / fluid),
    `sofa_relevant` (the organ-failure channels), and a frequency split by observation count.
    """
    import yaml
    if catalog_path is None:
        # cwd-relative default meant every type:*/sofa_relevant group vanished silently when the
        # script ran from anywhere but the repo root.
        catalog_path = Path(__file__).resolve().parents[1] / "10_scripts" / "features.yaml"
    try:
        cat = yaml.safe_load(open(catalog_path))
    except Exception:
        cat = {}
    obs = n.sum(axis=1)
    charted = obs > 0
    groups = {"all_charted": charted}
    types = {}
    for i, f in enumerate(features):
        t = (cat.get(str(f), {}) or {}).get("type")
        if t:
            types.setdefault(t, []).append(i)
    for t, idxs in types.items():
        m = np.zeros(len(features), dtype=bool); m[idxs] = True
        if (m & charted).sum() >= 3:
            groups[f"type:{t}"] = m & charted
    sofa = np.array([bool((cat.get(str(f), {}) or {}).get("sofa_relevant")) for f in features])
    if (sofa & charted).sum() >= 3:
        groups["sofa_relevant"] = sofa & charted
        groups["not_sofa_relevant"] = (~sofa) & charted
    # frequency tertiles among charted variables (dense vitals vs rare labs behave differently)
    if charted.sum() >= 6:
        q = np.quantile(obs[charted], [1 / 3, 2 / 3])
        groups["freq:low"] = charted & (obs <= q[0])
        groups["freq:mid"] = charted & (obs > q[0]) & (obs <= q[1])
        groups["freq:high"] = charted & (obs > q[1])
    return groups


def window_meta_features(x_in, mask_in, static, sofa=None, t0=None):
    """Per-window descriptors for patient clustering — INPUT-ONLY, by construction.

    Every field here is computable at inference time from the 24 h history: nothing touches
    `x_target`. That is not a detail, it is what makes the resulting router deployable (and
    keeps it honest — clustering on the target would leak the answer into the routing rule).

    Deliberately cheap and interpretable: how much is being measured (acuity proxy), how
    variable the signals are (stability), overall level, illness severity, and stay phase.
    """
    obs = mask_in.astype(np.float32)
    dens = obs.mean(axis=(1, 2))                                  # measurement intensity
    n_ch = (obs.sum(axis=1) > 0).mean(axis=1)                     # breadth of monitoring
    xz = np.where(mask_in, np.nan_to_num(x_in), np.nan)
    with np.errstate(invalid="ignore"):
        lvl = np.nanmean(xz, axis=(1, 2))
        vol = np.nanmean(np.nanstd(xz, axis=1), axis=1)           # within-window volatility
    feats = [dens, n_ch, np.nan_to_num(lvl), np.nan_to_num(vol)]
    if sofa is not None:
        s = np.asarray(sofa, dtype=np.float32)
        tot = s[..., -1] if s.ndim == 3 else s
        feats += [np.nan_to_num(tot.max(axis=1)), np.nan_to_num(tot[:, -1] - tot[:, 0])]
    if t0 is not None:
        feats.append(np.log1p(np.nan_to_num(np.asarray(t0, dtype=np.float32))))
    if static is not None and static.shape[1]:
        feats += [np.nan_to_num(static[:, j]) for j in range(min(static.shape[1], 4))]
    names = ["obs_density", "channels_monitored", "signal_level", "volatility"]
    if sofa is not None:
        names += ["sofa_max", "sofa_delta"]
    if t0 is not None:
        names += ["log_stay_hour"]
    if static is not None and static.shape[1]:
        names += [f"static{j}" for j in range(min(static.shape[1], 4))]
    M = np.stack(feats, axis=1).astype(np.float32)
    mu, sd = M.mean(axis=0), M.std(axis=0) + 1e-6
    return (M - mu) / sd, (mu, sd, names)


def describe_clusters(km, meta_stats, cstats, model_names, min_support: int = 200):
    """Interpretable centroids + the per-cluster model ranking.

    The performance question ("does routing per cluster beat one global rule?") is answered by
    the control row. This answers the mechanistic one: do the clusters differ in ways that
    change WHICH architecture wins? If the same model tops every cluster, per-cluster routing
    cannot help however the weights are computed — the partition is real but architecturally
    irrelevant.
    """
    mu, sd, names = meta_stats
    cent = km.cluster_centers_ * sd + mu           # back to original units
    sse, n = cstats["sse"], cstats["n"]
    K = sse.shape[0]
    out = {"names": names, "centroids": cent, "winners": [], "top3": []}
    for k in range(K):
        ok = n[k] >= min_support
        if ok.sum() == 0:
            out["winners"].append(None); out["top3"].append([]); continue
        per_model = sse[k][ok].sum(axis=0) / max(n[k][ok].sum(), 1)
        order = np.argsort(per_model)
        out["winners"].append(model_names[order[0]])
        out["top3"].append([(model_names[i], float(per_model[i])) for i in order[:3]])
    return out


def physio_features(x_in, mask_in, sofa=None, top_n: int = 24, chan_idx=None):
    """Cluster features from the OBSERVED measurements only — the patient's physiological state.

    Motivation (MF): the `meta` feature set clusters partly on *missingness* and on statics, and
    the statics carry an imputation artefact — UMM's 301-window cluster is simply "weight/height
    missing" (centroid weight 23.2, height 0.0), and HiRID charts no height at all, so those
    clusters describe data provenance rather than a patient phenotype.

    Here each window is represented by its **mean observed z-value per channel**, restricted to
    the `top_n` best-covered channels (vitals and common labs), plus SOFA. No statics, no
    density, no channel-count — so a cluster can only form around what was actually measured.

    Residual caveat, unavoidable at 11-18 % observation density: a channel unobserved in a given
    window is filled with 0, which in z-space is the population mean. Restricting to the
    best-covered channels keeps that fill rare where it would matter most.
    """
    obs = mask_in.astype(np.float32)
    if chan_idx is None:
        cov = obs.mean(axis=(0, 1))
        chan_idx = np.argsort(-cov)[:top_n]
    num = (np.nan_to_num(x_in) * obs)[:, :, chan_idx].sum(axis=1)
    den = obs[:, :, chan_idx].sum(axis=1)
    ch_mean = np.where(den > 0, num / np.clip(den, 1e-6, None), 0.0)
    feats = [ch_mean]
    names = [f"ch{j}" for j in chan_idx]
    if sofa is not None:
        s = np.asarray(sofa, dtype=np.float32)
        tot = s[..., -1] if s.ndim == 3 else s
        feats.append(np.stack([np.nan_to_num(tot.max(axis=1)),
                               np.nan_to_num(tot[:, -1] - tot[:, 0])], axis=1))
        names += ["sofa_max", "sofa_delta"]
    M = np.concatenate(feats, axis=1).astype(np.float32)
    mu, sd = M.mean(axis=0), M.std(axis=0) + 1e-6
    return (M - mu) / sd, (mu, sd, names), chan_idx


def error_features(predictors, idx, x_in, mask_in, static, x_target, mask_target, device,
                   chunk: int = 2048, raw: bool = False):
    """ORACLE cluster features: each window's per-model error profile.

    This asks MF's question directly — "is there a group of patients where Autoformer predicts HR
    well and Informer does not?" — by clustering windows in the space of *which model was right*,
    normalised per window so the clusters capture the SHAPE of the model ranking rather than how
    hard the window was.

    IT USES THE TARGETS, so it is **not deployable**: a real router must assign clusters from the
    input alone. Its value is as an UPPER BOUND. If clustering directly on the answers still does
    not yield materially different per-cluster winners and a real gain, then no input-based
    router — however clever its features — can succeed, and Tier 4 is closed. If it does, the gap
    between it and the `physio` router measures what better routing features could still buy.
    """
    M = len(predictors)
    out = np.zeros((len(idx), M), dtype=np.float32)
    for s in range(0, len(idx), chunk):
        sub = idx[s:s + chunk]
        y = np.nan_to_num(x_target[sub]).astype(np.float32)
        m = mask_target[sub].astype(np.float32)
        P = np.stack([p.predict(sub, x_in, mask_in, static, device) for p in predictors], axis=-1)
        P = np.nan_to_num(P) * m[..., None]
        se = ((P - (y * m)[..., None]) ** 2).sum(axis=(1, 2))          # (n, M)
        cnt = np.clip(m.sum(axis=(1, 2)), 1, None)[:, None]
        out[s:s + len(sub)] = se / cnt
    # normalise per window: cluster on the RANKING shape, not on window difficulty
    rel = out / np.clip(out.mean(axis=1, keepdims=True), 1e-8, None)
    mu, sd = rel.mean(axis=0), rel.std(axis=0) + 1e-6
    names = [p.name[:13] for p in predictors]
    if raw:
        # caller standardises once and reuses the statistics across halves — scoring each half
        # with its own mu/sd would place them in different coordinate systems
        return rel, (mu, sd, names)
    return (rel - mu) / sd, (mu, sd, names)


def fit_router(err_fit, X_fit, X_ev, K: int, seed: int = 42):
    """The DEPLOYABLE two-stage router (MF's design, closed loop).

    Stage 1 — on the FIT half only, cluster windows by their per-model error profile. Using the
    targets here is legitimate: this half exists to fit things.
    Stage 2 — train `input -> cluster` on the fit half, then assign every EVAL-half window from
    its INPUT ALONE. The eval half never contributes a target to the routing decision, so what
    comes out is what a deployed system would actually achieve — unlike the oracle variant, whose
    eval labels were also derived from the answers and which therefore only bounds the gain.

    Returns (fit labels, predicted eval labels, diagnostics).
    """
    from sklearn.ensemble import RandomForestClassifier
    km = fit_clusters(err_fit, K, seed)
    c_fit = km.predict(err_fit)
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                 random_state=seed, class_weight="balanced")
    clf.fit(X_fit, c_fit)
    c_ev = clf.predict(X_ev)
    return c_fit, c_ev, dict(
        fit_sizes=[int((c_fit == k).sum()) for k in range(K)],
        pred_sizes=[int((c_ev == k).sum()) for k in range(K)],
        train_acc=float((clf.predict(X_fit) == c_fit).mean()))


def cluster_context_test(lab_fit, lab_ev, X_fit, X_ev, meta_stats, K, seed: int = 42):
    """Clusters defined on the TARGET side — do they have any INPUT signature? (MF's question.)

    Two-stage design: partition windows by *which model wins* (error space, an oracle), then come
    back to the input and ask whether those groups are recognisable from what a deployed system
    would actually see. This separates two very different failure modes that the plain Tier-4
    result conflates:

      * the clusters are not identifiable from the input  -> no CONTEXT exists; no router can
        ever route, however good its features. Tier 4 is closed.
      * the clusters ARE identifiable but routing still does not pay -> context exists but the
        per-cluster differences are too small/unstable to exploit. Better features would not
        help either, but for a different reason.

    Returns the input-space centroids of the error clusters (in original units) plus the balanced
    accuracy of an input -> cluster classifier, fitted on the fit half and scored on the disjoint
    eval half, against the 1/K chance baseline.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score

    mu, sd, names = meta_stats
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=20, n_jobs=-1,
                                 random_state=seed, class_weight="balanced")
    clf.fit(X_fit, lab_fit)
    pred = clf.predict(X_ev)
    bacc = float(balanced_accuracy_score(lab_ev, pred))
    cent = np.stack([X_fit[lab_fit == k].mean(axis=0) if (lab_fit == k).any()
                     else np.zeros(X_fit.shape[1]) for k in range(K)])
    cent_orig = cent * sd + mu
    # which input features carry the signal at all
    imp = sorted(zip(names, clf.feature_importances_), key=lambda t: -t[1])[:6]
    return dict(balanced_accuracy=bacc, chance=1.0 / K, centroids=cent_orig,
                names=names, top_features=imp,
                sizes=[int((lab_fit == k).sum()) for k in range(K)])


def fit_clusters(meta_fit, k: int = 6, seed: int = 42):
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(meta_fit)
    return km


def accumulate_cluster_stats(predictors, idx, x_in, mask_in, static, x_target, mask_target,
                             cluster_of, K, device, chunk: int = 2048):
    """Per-(cluster, feature) statistics with hours POOLED.

    Hours are pooled deliberately: the per-hour axis was measured to buy nothing, and pooling
    keeps enough observations per (cluster, feature) cell to rank models meaningfully. Splitting
    K ways is already a big cut in support — this is where §13.6a's noise trap would reappear.
    """
    M = len(predictors)
    F = x_target.shape[2]
    G = np.zeros((K, F, M, M)); b = np.zeros((K, F, M))
    sse = np.zeros((K, F, M)); n = np.zeros((K, F)); yy = np.zeros((K, F))
    for s in range(0, len(idx), chunk):
        sub = idx[s:s + chunk]
        c = cluster_of[s:s + chunk]
        y = np.nan_to_num(x_target[sub]).astype(np.float32)
        m = mask_target[sub].astype(np.float32)
        P = np.stack([p.predict(sub, x_in, mask_in, static, device) for p in predictors], axis=-1)
        P = np.nan_to_num(P) * m[..., None]
        ym = y * m
        for kk in range(K):
            sel = c == kk
            if not sel.any():
                continue
            Pk, yk, mk = P[sel], ym[sel], m[sel]
            G[kk] += np.einsum("nhfi,nhfj->fij", Pk, Pk, optimize=True)
            b[kk] += np.einsum("nhfi,nhf->fi", Pk, yk, optimize=True)
            sse[kk] += ((Pk - yk[..., None]) ** 2).sum(axis=(0, 1))
            n[kk] += mk.sum(axis=(0, 1))
            yy[kk] += (yk ** 2).sum(axis=(0, 1))
    return dict(G=G, b=b, sse=sse, n=n, yy=yy)


def cluster_ranking_weights(cstats, global_var_weights, min_support: int, top_k: int = 5,
                            scheme: str = "harmonic"):
    """Ranking-only weights per (cluster, feature); falls back to the GLOBAL per-variable
    weights wherever a cluster cell is too thin to rank reliably.

    `scheme='harmonic'` = 1, 1/2, ... 1/k over the best k models in that cluster;
    `scheme='inv_mse'`  = weights proportional to 1/MSE over all models in that cluster.
    Neither estimates a coefficient, so the extra cluster axis cannot inflate variance the way a
    per-cluster ridge would.
    """
    sse, n = cstats["sse"], cstats["n"]
    K, F, M = sse.shape
    W = np.zeros((K, F, M))
    per = sse / np.clip(n, 1, None)[..., None]
    if scheme == "harmonic":
        top_k = min(top_k, M)             # same guard as build_weight_sets
        hw = 1.0 / np.arange(1, top_k + 1); hw /= hw.sum()
    for kk in range(K):
        for f in range(F):
            if n[kk, f] < min_support:
                W[kk, f] = global_var_weights[f]
                continue
            if scheme == "harmonic":
                rank = np.argsort(per[kk, f])[:top_k]
                W[kk, f, rank] = hw
            else:
                inv = 1.0 / np.clip(per[kk, f], 1e-8, None)
                W[kk, f] = inv / inv.sum()
    return W


def score_cluster_weights(W, cstats):
    """SSE of per-(cluster, feature) weights, aggregated to a per-variable MSE."""
    G, b, yy, n = cstats["G"], cstats["b"], cstats["yy"], cstats["n"]
    quad = np.einsum("kfi,kfij,kfj->kf", W, G, W, optimize=True)
    lin = np.einsum("kfi,kfi->kf", W, b, optimize=True)
    sse = quad - 2.0 * lin + yy
    per_feat = sse.sum(axis=0) / np.clip(n.sum(axis=0), 1, None)
    keep = n.sum(axis=0) > 0
    return dict(mse_per_var=float(per_feat[keep].mean()),
                mse_global=float(sse.sum() / max(n.sum(), 1)),
                per_feature=per_feat)


def ambiguity_decomposition(stats: dict):
    """Krogh & Vedelsby: for squared loss the uniform ensemble error decomposes EXACTLY as

        E_ens  =  (1/M) Sum_m E_m  -  (1/M) Sum_m E[(f_m - f_ens)^2]
                  \\____ mean individual ____/   \\______ ambiguity (diversity) ______/

    This is an identity, not a bound, so it converts "how orthogonal are the architectures?"
    into an MSE quantity in the SAME units as the forecast error — i.e. it says how much
    ensembling can possibly buy, per (variable, hour) cell, before any combiner is fitted.

    Both terms come out of the sufficient statistics with no extra inference:
        Sum_n f_m^2       = G[m,m]
        Sum_n f_m f_ens   = (1/M) Sum_m' G[m,m']
        Sum_n f_ens^2     = (1/M^2) Sum_{m,m'} G[m,m']
    =>  ambiguity * n     = (1/M) tr(G) - (1/M^2) Sum_{m,m'} G[m,m']
    """
    G, sse, n = stats["G"], stats["sse"], stats["n"]
    M = sse.shape[-1]
    denom = np.clip(n, 1, None)
    tr = np.einsum("fhmm->fh", G)
    tot = G.sum(axis=(2, 3))
    ambiguity = (tr / M - tot / (M * M)) / denom
    mean_ind = sse.mean(axis=2) / denom
    return dict(mean_individual=mean_ind, ambiguity=ambiguity,
                ensemble_pred=mean_ind - ambiguity, n=n)


def residual_correlation(stats: dict):
    """Mean pairwise residual correlation per (variable, hour), from the same statistics.

        Sum_n r_m r_m' = G[m,m'] - b[m] - b[m'] + Sum y^2      (r_m = f_m - y)

    Uncentered (cosine) correlation: residuals are near zero-mean, so this matches the
    §13.5 global figure while resolving it per cell — the point being to find WHERE the
    architectures decorrelate, since that is the only place a combiner can pay.
    """
    G, b, yy, n = stats["G"], stats["b"], stats["yy"], stats["n"]
    M = b.shape[-1]
    C = G - b[..., :, None] - b[..., None, :] + yy[..., None, None]
    d = np.sqrt(np.clip(np.einsum("fhmm->fhm", C), 1e-12, None))
    R = C / (d[..., :, None] * d[..., None, :])
    off = (R.sum(axis=(2, 3)) - np.einsum("fhmm->fh", R)) / (M * (M - 1))
    return dict(mean_pairwise=off, full=R, n=n)


def score_weights(W: dict, stats_eval: dict):
    """Closed-form SSE of any weight tensor on the eval half, from its sufficient statistics.

        SSE(w) = w' G w - 2 w' b + Sum y^2
    """
    G, b, yy, n = stats_eval["G"], stats_eval["b"], stats_eval["yy"], stats_eval["n"]
    out = {}
    for name, w in W.items():
        quad = np.einsum("fhi,fhij,fhj->fh", w, G, w, optimize=True)
        lin = np.einsum("fhi,fhi->fh", w, b, optimize=True)
        sse = quad - 2.0 * lin + yy
        per_feat = sse.sum(axis=1) / np.clip(n.sum(axis=1), 1, None)
        keep = n.sum(axis=1) > 0
        out[name] = dict(mse_per_var=float(per_feat[keep].mean()),
                         mse_global=float(sse.sum() / max(n.sum(), 1)),
                         per_feature=per_feat, evaluable_features=int(keep.sum()))
    return out
