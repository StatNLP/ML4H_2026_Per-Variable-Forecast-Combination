"""
10_scripts/analyze_error_structure.py
Error-structure analysis for the forecasting models: (1) how forecast error
grows over the prediction horizon, and (2) which variables drive the error.

Reads every checkpoints/*/final_metrics.json (written by harmonize.train_forecast),
which stores per-run:
    val.horizon_mse  — list of 24 MSE values, one per forecast hour
    val.feat_mse     — {feature: MSE} over the validation set (z-scored units)
    val.mse_per_var  — scalar used to pick the best run per (dataset, model)
    combo.train_h5   — to recover the dataset name

For each (dataset, model bucket) the BEST run (lowest mse_per_var) is used.

Outputs (under --out-dir, default docs/):
    tables/horizon_mse.csv          (dataset, model, h1..h24, ratio_h24_h1)
    tables/feature_mse.csv          (dataset, model, feature, mse)
    tables/feature_mse_ranked.csv   (model, feature, mse_mean_over_datasets, rank)
    tables/feature_mse_sofa_vars.csv (SOFA-relevant subset of the ranking; folded in from the
                                      retired analyze_feature_performance.py)
    figures/horizon_degradation.png (one panel per dataset; line per model)
    figures/feature_mse_<model>.png (hardest/easiest features, mean over datasets)
    tables/arch_feature_mse.csv     (architecture x feature val MSE, mean over datasets)
    tables/arch_feature_mse_relative.csv (same / per-feature best architecture)
    figures/arch_feature_heatmap.png (architecture x feature heatmap, absolute + relative)

Usage:
    python 10_scripts/analyze_error_structure.py
    python 10_scripts/analyze_error_structure.py --models denseAR strats_hourly strats_raw persistence
    python 10_scripts/analyze_error_structure.py --no-figures
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["mimic_iii", "mimic_iv", "eicu", "hirid", "umm"]

# Display name for the private cohort. The submission is anonymous, so the artwork must not carry
# the centre's initials either -- a reviewer sees the rendered figure, not just the LaTeX. Set this
# to the real name for the camera-ready and re-run; nothing else needs to change.
COHORT = "UMM"

# model bucket order (also the plotting order / legend order)
MODEL_ORDER = ["null", "persistence", "autoformer", "dlinear",
               "strats_hourly", "strats_raw", "dense_dms", "informer",
               "gru", "lstm", "mamba", "patchtst", "denseAR",
               "chronos", "chronos2", "moirai", "tirex2", "mira"]

# Foundation-model buckets are drawn dashed (so the FMs are visually separable from the
# in-domain models in the horizon figure). PRETTY = legible legend labels.
FM_BUCKETS = {"chronos", "chronos2", "moirai", "tirex2", "mira"}
PRETTY = {"null": "null", "persistence": "LOCF", "denseAR": "DenseAR",
          "dense_dms": "DenseDMS", "informer": "Informer", "autoformer": "Autoformer",
          "dlinear": "DLinear", "strats_hourly": "STraTS", "strats_raw": "STraTS-raw",
          "gru": "GRU", "lstm": "LSTM", "mamba": "Mamba", "patchtst": "PatchTST",
          "chronos": "Chronos-Bolt", "chronos2": "Chronos-2",
          "moirai": "Moirai", "tirex2": "TiRex-2",
          "mira": "MIRA"}

# Foundation-model / precomputed eval JSONs live in docs/tables/eval_*.json (NOT checkpoints/),
# with a different schema: {dataset, model, forecast:{val_mse_per_var, horizon_mse, feat_mse}}.
# We fold the best variant per FM family (zero-shot vs fine-tuned) into the same best[(ds,bucket)]
# map so the cross-model horizon / per-feature tables include the FMs. NB: MIRA is evaluated on a
# 6000-window subsample (§9.7), the others on the full test — the horizon *shape* / ratio_h24_h1 is
# the robust cross-model comparison; read MIRA's absolute level with that caveat.
# (chronos2 before chronos so the more specific prefix matches first.)
FM_PREFIX = [("eval_chronos2_", "chronos2"), ("eval_chronos_", "chronos"),
             ("eval_moirai_", "moirai"), ("eval_tirex2_", "tirex2"),
             ("eval_mira_", "mira")]


def bucket(run_name: str, model: str) -> str | None:
    if run_name.startswith("baseline_null"):        return "null"
    if run_name.startswith("baseline_persistence"): return "persistence"
    if run_name.startswith("strats_raw_"):          return "strats_raw"
    if run_name.startswith("strats_"):              return "strats_hourly"
    if run_name.startswith("hpsf_"):                return "denseAR"
    if model == "dense_dms":                        return "dense_dms"
    if model == "informer":                         return "informer"
    if model == "autoformer":                       return "autoformer"
    if model == "dlinear":                          return "dlinear"
    if model == "gru":                              return "gru"
    if model == "lstm":                             return "lstm"
    if model == "mamba":                            return "mamba"
    if model == "patchtst":                         return "patchtst"
    if model == "dense_autoregressive":             return "denseAR"
    return None


def dataset_of(combo: dict) -> str | None:
    m = re.match(r"([a-z_]+)_seed\d+", Path(combo.get("train_h5", "")).stem)
    return m.group(1) if m else None


def scan(ckpt_dir: Path) -> dict:
    """Return best[(dataset, modelbucket)] = val-dict (lowest mse_per_var)."""
    best: dict[tuple, dict] = {}
    best_mse: dict[tuple, float] = {}
    for jf in ckpt_dir.glob("*/final_metrics.json"):
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        v = d.get("val", {})
        combo = d.get("combo", {})
        mpv = v.get("mse_per_var")
        if mpv is None:
            continue
        ds = dataset_of(combo)
        mb = bucket(jf.parent.name, combo.get("model", ""))
        if ds is None or mb is None:
            continue
        key = (ds, mb)
        if key not in best_mse or mpv < best_mse[key]:
            best_mse[key] = mpv
            best[key] = v
    return best


def scan_fm(eval_dir: Path) -> dict:
    """Best FM variant per (dataset, family) from docs/tables/eval_{chronos,chronos2,moirai,mira}_*.json,
    normalised to the same {mse_per_var, horizon_mse, feat_mse} schema as scan()."""
    best: dict[tuple, dict] = {}
    best_mse: dict[tuple, float] = {}
    for jf in sorted(eval_dir.glob("eval_*.json")):
        mb = next((b for pre, b in FM_PREFIX if jf.name.startswith(pre)), None)
        if mb is None:
            continue
        try:
            d = json.load(open(jf))
        except Exception:
            continue
        fc = d.get("forecast", {}); ds = d.get("dataset")
        mpv = fc.get("val_mse_per_var")
        if mpv is None or ds not in DATASETS:
            continue
        key = (ds, mb)
        if key not in best_mse or mpv < best_mse[key]:
            best_mse[key] = mpv
            best[key] = {"mse_per_var": mpv, "horizon_mse": fc.get("horizon_mse"),
                         "feat_mse": fc.get("feat_mse")}
    return best


def build_horizon_table(best: dict, models: list[str]) -> pd.DataFrame:
    rows = []
    for ds in DATASETS:
        for mb in models:
            v = best.get((ds, mb))
            hm = v.get("horizon_mse") if v else None
            if not hm:
                continue
            row = {"dataset": ds, "model": mb}
            row.update({f"h{i+1}": round(float(hm[i]), 4) for i in range(len(hm))})
            row["ratio_h24_h1"] = round(hm[-1] / hm[0], 3) if hm[0] else float("nan")
            rows.append(row)
    return pd.DataFrame(rows)


def build_feature_tables(best: dict, models: list[str]):
    long_rows = []
    for ds in DATASETS:
        for mb in models:
            v = best.get((ds, mb))
            fm = v.get("feat_mse") if v else None
            if not fm:
                continue
            for feat, mse in fm.items():
                long_rows.append({"dataset": ds, "model": mb,
                                  "feature": feat, "mse": round(float(mse), 4)})
    long_df = pd.DataFrame(long_rows)
    # ranked: mean over datasets, per model
    if long_df.empty:
        return long_df, pd.DataFrame()
    # Never-charted channels arrive as an exact 0.0 placeholder, so a plain mean over datasets
    # divides a real error by the number of datasets and makes a channel that ONE database records
    # look like the easiest in the panel: denseAR/mpv is eICU-only at 0.1963 and ranked first at
    # 0.1963/5 = 0.0393. Average over the datasets that actually chart the channel.
    _lf = long_df[long_df["mse"] > 0]
    ranked = (_lf.groupby(["model", "feature"])["mse"]
              .mean().reset_index().rename(columns={"mse": "mse_mean_over_datasets"}))
    ranked["mse_mean_over_datasets"] = ranked["mse_mean_over_datasets"].round(4)
    ranked["rank"] = ranked.groupby("model")["mse_mean_over_datasets"].rank("dense").astype(int)
    ranked = ranked.sort_values(["model", "mse_mean_over_datasets"])
    return long_df, ranked


def plot_horizon(htab: pd.DataFrame, models: list[str], out_png: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    hcols = [c for c in htab.columns if re.fullmatch(r"h\d+", c)]
    hours = [int(c[1:]) for c in hcols]
    n = len(DATASETS)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    axes = axes.ravel()
    for ax, ds in zip(axes, DATASETS):
        sub = htab[htab["dataset"] == ds]
        for mb in models:
            r = sub[sub["model"] == mb]
            if r.empty:
                continue
            ls = "--" if mb in FM_BUCKETS else "-"
            lw = 1.6 if mb in FM_BUCKETS else 1.2
            ax.plot(hours, r[hcols].values.ravel(), marker=".", linestyle=ls,
                    linewidth=lw, label=PRETTY.get(mb, mb))
        ax.set_title(ds); ax.set_xlabel("forecast hour"); ax.set_ylabel("val MSE")
        ax.grid(alpha=0.3)
    axes[len(DATASETS)].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[len(DATASETS)].legend(h, l, loc="center", title="model  (FM = dashed)", fontsize=8)
    fig.suptitle("Forecast error vs. horizon (val MSE per hour)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def plot_features(ranked: pd.DataFrame, model: str, out_png: Path, top: int = 15):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sub = ranked[ranked["model"] == model].sort_values("mse_mean_over_datasets")
    if sub.empty:
        return
    easiest = sub.head(top); hardest = sub.tail(top)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 6))
    a1.barh(easiest["feature"], easiest["mse_mean_over_datasets"], color="#3a7")
    a1.set_title(f"Easiest {top} (lowest MSE)"); a1.invert_yaxis(); a1.grid(alpha=0.3, axis="x")
    a2.barh(hardest["feature"], hardest["mse_mean_over_datasets"], color="#c44")
    a2.axvline(1.0, color="k", ls="--", lw=0.8, label="= per-feature mean (null)")
    a2.set_title(f"Hardest {top} (highest MSE)"); a2.invert_yaxis(); a2.grid(alpha=0.3, axis="x"); a2.legend()
    fig.suptitle(f"Per-variable val MSE — {model} (mean over datasets, z-scored units)")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def build_arch_feature_matrix(long_df: pd.DataFrame, models: list[str]):
    """(architecture x feature) val-MSE matrix, mean over datasets.

    Two corrections matter for this table to be honest:
      * `feat_mse` carries **0.0 placeholders** for channels a dataset never charts (the 94-col
        union tensor is shared, unavailable columns are masked). Those are not "perfect
        predictions" — they are absent, and averaging them in would reward whichever model
        happened to be run on the sparsest dataset. Dropped.
      * A (dataset, feature) cell is kept only where **every** selected architecture has a
        value, so each column compares like with like. Without this a model missing on one
        dataset gets a mean over a different, possibly easier, dataset mix.

    Returns (abs_df, rel_df): absolute mean MSE, and MSE / per-feature best-architecture MSE
    (1.0 = this architecture is the best on that variable). The relative view is the one that
    answers "do architectures actually differ per variable?" — absolute MSE is dominated by
    how intrinsically noisy each variable is, which is a property of the variable, not the model.
    """
    df = long_df[long_df["mse"] > 0].copy()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df = df[df["model"].isin(models)]
    # keep only (dataset, feature) cells covered by every selected architecture
    n_models = df["model"].nunique()
    full = (df.groupby(["dataset", "feature"])["model"].nunique() == n_models)
    keep = set(full[full].index)
    df = df[[(d, f) in keep for d, f in zip(df["dataset"], df["feature"])]]
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    abs_df = df.pivot_table(index="model", columns="feature", values="mse", aggfunc="mean")
    abs_df = abs_df.reindex([m for m in models if m in abs_df.index])
    # hardest -> easiest, so the figure reads left to right
    abs_df = abs_df[abs_df.mean(axis=0).sort_values(ascending=False).index]
    rel_df = abs_df.div(abs_df.min(axis=0), axis=1)
    return abs_df, rel_df


def plot_arch_feature_heatmap(abs_df: pd.DataFrame, rel_df: pd.DataFrame, out_png: Path,
                              max_features: int | None = None):
    """Two stacked heatmaps: absolute per-variable MSE and MSE relative to the per-variable best."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if abs_df.empty:
        return
    a, r = abs_df, rel_df
    if max_features:
        a = a.iloc[:, :max_features]; r = r.iloc[:, :max_features]
    nfeat, nmod = a.shape[1], a.shape[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, nfeat * 0.22), max(6, nmod * 0.62)))

    vmax = float(np.nanpercentile(a.values, 97))   # clip so a few wild variables do not flatten it
    im1 = ax1.imshow(a.values, aspect="auto", cmap="magma_r", vmin=0, vmax=vmax)
    ax1.set_title(f"Per-variable val MSE (z-scored units, mean over datasets) — "
                  f"{nmod} architectures x {nfeat} variables, hardest left")
    fig.colorbar(im1, ax=ax1, fraction=0.015, pad=0.006, label=f"MSE (clipped at p97={vmax:.2f})")

    im2 = ax2.imshow(r.values, aspect="auto", cmap="RdBu_r", vmin=1.0, vmax=2.0)
    ax2.set_title("Same, RELATIVE to the best architecture per variable "
                  "(1.0 = best; blue = near-best, red = >=2x worse)")
    fig.colorbar(im2, ax=ax2, fraction=0.015, pad=0.006, label="MSE / best-per-variable MSE")

    for ax, m in ((ax1, a), (ax2, r)):
        ax.set_yticks(range(len(m.index)))
        ax.set_yticklabels([PRETTY.get(i, i) for i in m.index], fontsize=8)
        ax.set_xticks(range(len(m.columns)))
        ax.set_xticklabels(m.columns, rotation=90, fontsize=5.5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def plot_arch_feature_ranked(abs_df: pd.DataFrame, out_png: Path, max_features: int | None = None,
                             font_scale: float = 1.0, pool_only: bool = False,
                             height_scale: float = 1.0, easiest: int = 0,
                             model_font_scale: float = 1.0):
    """models x features, coloured by RANK, with both marginals.

    Why rank and not MSE: absolute per-variable MSE is dominated by how intrinsically noisy each
    VARIABLE is, so an MSE-coloured map mostly redraws the column structure and hides the thing
    actually asked for — *which model is good where*. Ranking within each feature column removes
    the variable's difficulty and leaves only the model comparison.

    The two marginals carry what the rank map deliberately throws away:
      * top strip   — mean MSE over all models per feature = how hard that variable is
                      (this is the "average over all models in one view").
      * right strip — mean MSE over all features per model = the overall ladder position.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if abs_df.empty:
        return
    # The table carries every architecture ever evaluated; the paper's pool is 14. Showing the
    # other four (Chronos, Chronos-2, MIRA, the triplet-native STraTS) invites "why are these not
    # in your pool" from a reader who can now read the row labels.
    POOL = ["null", "persistence", "dlinear", "dense_dms", "denseAR", "gru", "lstm", "mamba",
            "patchtst", "informer", "autoformer", "strats_hourly", "moirai", "tirex2"]
    if pool_only:
        keep = [m for m in POOL if m in abs_df.index]
        missing = [m for m in POOL if m not in abs_df.index]
        if missing:
            print(f"[ranked: pool members absent from the table: {missing}]")
        abs_df = abs_df.loc[keep]
    # Re-sort the columns by the difficulty ACTUALLY PLOTTED. The saved table is ordered by the
    # mean over all evaluated models; pool_only then drops four of them, so the stored order no
    # longer matches the bars and the top strip came out non-monotone (bilirubin_direct sat left
    # of two variables with higher plotted MSE).
    abs_df = abs_df[abs_df.mean(axis=0).sort_values(ascending=False).index]
    # The right marginal is each model's mean over EVERY variable, not over the subset drawn, so
    # truncating the columns cannot move a model up or down the ladder.
    model_qual = abs_df.mean(axis=1).sort_values()
    # `easiest` appends the N easiest variables after the hardest block. The hard tail is where the
    # models disagree, but a reader cannot tell whether the pattern is specific to hard channels
    # without seeing the other end; the two blocks are separated by a rule.
    if max_features and easiest:
        a = pd.concat([abs_df.iloc[:, :max_features], abs_df.iloc[:, -easiest:]], axis=1)
        split_at = max_features
    else:
        a = abs_df if not max_features else abs_df.iloc[:, :max_features]
        split_at = None
    rank = a.rank(axis=0, method="min")                 # 1 = best model on that feature
    feat_diff = a.mean(axis=0)                          # per-feature difficulty
    rank = rank.loc[model_qual.index]                   # order rows best -> worst overall
    a = a.loc[model_qual.index]
    nF, nM = a.shape[1], a.shape[0]

    # height_scale shrinks the figure vertically without touching the width, so the horizontal
    # scale LaTeX applies (and with it the rendered font size) is unchanged.
    fig = plt.figure(figsize=(max(13, nF * 0.24), max(7, nM * 0.62) * height_scale))
    gs = fig.add_gridspec(2, 2, width_ratios=[nF * 0.24, 2.6], height_ratios=[1.05, nM * 0.55],
                          wspace=0.02, hspace=0.03)
    ax_top, ax_main, ax_right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    ax_top.bar(range(nF), feat_diff.values, color="#444", width=0.85)
    ax_top.set_xlim(-0.5, nF - 0.5); ax_top.set_xticks([])
    ax_top.set_ylabel("mean\nMSE", fontsize=7 * min(font_scale, 1.5), labelpad=2)
    for sp in ("top", "right", "left"):
        ax_top.spines[sp].set_visible(False)


    im = ax_main.imshow(rank.values, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=nM)
    ax_main.set_yticks(range(nM))
    # the row labels are the only text whose size is set by the reader's eye rather than by
    # how many of them there are, so they get their own multiplier
    ax_main.set_yticklabels([PRETTY.get(i, i) for i in rank.index],
                            fontsize=8 * font_scale * model_font_scale)
    ax_main.set_xticks(range(nF))
    ax_main.set_xticklabels(a.columns, rotation=90, fontsize=5 * font_scale)

    ax_right.barh(range(nM), model_qual.values, color="#444", height=0.8)
    ax_right.set_ylim(nM - 0.5, -0.5); ax_right.set_yticks([])
    ax_right.set_xlabel("mean MSE", fontsize=7 * min(font_scale, 1.5), labelpad=2)
    for sp in ("top", "right", "left"):
        ax_right.spines[sp].set_visible(False)
    for i, v in enumerate(model_qual.values):
        ax_right.text(v, i, f" {v:.3f}", va="center", fontsize=6 * font_scale)

    if split_at is not None:                            # mark the discontinuity in the x axis
        for ax in (ax_top, ax_main):
            ax.axvline(split_at - 0.5, color="k", lw=1.6 * min(font_scale, 1.6), zorder=5)

    cb = fig.colorbar(im, ax=ax_right, fraction=0.09, pad=0.35)
    cb.set_label("rank within variable (1 = best)", fontsize=7 * min(font_scale, 1.5))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def plot_arch_feature_colnorm(abs_df: pd.DataFrame, out_png: Path, max_features: int | None = None,
                              span: float = 0.45):
    """models x features, each column normalised so the COLUMN MEAN = 1.0.

    Three maps of the same matrix answer three different questions, and this is the third:
      * absolute MSE          -> how hard is each variable (column structure dominates)
      * MSE / per-variable best -> how far is each model from the winner (floors at 1.0, so all
                                 the good models pile up at the bottom of the scale)
      * **MSE / per-variable MEAN (here)** -> how does each model compare to the *typical* model
                                 on that variable, centred at 1.0 and symmetric in both directions

    Centring on the mean rather than the best is what makes small differences legible: the scale
    now spends its dynamic range on the spread *between* models instead of on the gap to a single
    outlier, and being better-than-typical is visible as its own colour rather than as
    "slightly above 1".

    Diverging colour, centred at 1.0: green = better than the typical model on that variable,
    dark red = worse. `span` sets the symmetric window (default +/-45 %).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if abs_df.empty:
        return
    a = abs_df if not max_features else abs_df.iloc[:, :max_features]
    norm = a.div(a.mean(axis=0), axis=1)                       # column mean == 1.0 by construction
    order = norm.mean(axis=1).sort_values().index              # best overall model first
    norm = norm.loc[order]
    nF, nM = norm.shape[1], norm.shape[0]

    fig, ax = plt.subplots(figsize=(max(13, nF * 0.24), max(6, nM * 0.6)))
    im = ax.imshow(norm.values, aspect="auto", cmap="RdYlGn_r",
                   vmin=1 - span, vmax=1 + span)
    ax.set_yticks(range(nM))
    ax.set_yticklabels([f"{PRETTY.get(i, i)}   [{norm.loc[i].mean():.2f}]" for i in norm.index],
                       fontsize=8)
    ax.set_xticks(range(nF))
    ax.set_xticklabels(norm.columns, rotation=90, fontsize=5)
    ax.set_title("Per-variable MSE relative to the AVERAGE model for that variable "
                 "(column mean = 1.00)\n"
                 "green = better than typical · dark red = worse · [ ] = model's mean over variables",
                 fontsize=11)
    cb = fig.colorbar(im, ax=ax, fraction=0.012, pad=0.006)
    cb.set_label("MSE / column mean", fontsize=8)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)
    return norm


def plot_pool_corr(csv_path: Path, out_png: Path):
    """Member x member residual-correlation heatmap (the matrix behind the pool-diversity
    table). Validation statistics -- it is computed from the same cached Gram matrices the
    combiner is fitted on, so it describes the fitting split, not test."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = pd.read_csv(csv_path, index_col=0)
    names = list(M.columns)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    im = ax.imshow(M.values, cmap="RdBu_r", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            v = M.values[i, j]
            ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.5,
                    color="white" if v > 0.88 or v < 0.52 else "black")
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="mean pairwise residual correlation")
    ax.set_title("Lower = more complementary (validation)", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def plot_pool_size(out_png: Path, tables_dir: Path | None = None):
    """Gain against the best single model as a function of POOL SIZE.

    Random subsets of the pool, five draws per size, combined with the shrunk simplex and
    scored under the development protocol (fit on one stay-grouped half of validation, score
    on the other), read from featstack_<ds>_val_rb_kabl_k<k>_d<d>_by_group.csv. The full-pool
    point comes from featstack_<ds>_val_by_group.csv. Lines are the mean over the five draws;
    the individual draws are drawn faint, because the only subsets that lose to the best single
    model are single draws at k=2 and the mean hides them.
    """
    import re, glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    td = tables_dir or (ROOT / "docs" / "tables")

    def _delta(path):
        d = pd.read_csv(path)
        r = d[d.group == "all_charted"]
        if r.empty:
            return None
        r = r.iloc[0]
        return 100.0 * (r["simplex_shrunk500_var"] - r["best_single"]) / r["best_single"]

    draws = defaultdict(list)
    for f in glob.glob(str(td / "featstack_*_val_rb_kabl_k*_d*_by_group.csv")):
        m = re.search(r"featstack_(\w+?)_val_rb_kabl_k(\d+)_d(\d+)_by_group", f)
        v = _delta(f)
        if m and v is not None:
            draws[(m.group(1), int(m.group(2)))].append(v)
    full = {}
    for ds in DATASETS:
        f = td / f"featstack_{ds}_val_by_group.csv"
        if f.exists():
            v = _delta(f)
            if v is not None:
                full[ds] = v

    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    for ds in DATASETS:
        ks = sorted({k for (d, k) in draws if d == ds})
        if not ks:
            continue
        line, = ax.plot(ks, [float(np.mean(draws[(ds, k)])) for k in ks],
                        marker="o", ms=4, lw=1.6,
                        label=COHORT if ds == "umm" else PRETTY.get(ds, ds))
        for k in ks:
            ax.plot([k] * len(draws[(ds, k)]), draws[(ds, k)], ".",
                    ms=2.5, alpha=0.30, color=line.get_color())
        if ds in full:
            ax.plot([14], [full[ds]], marker="*", ms=10, color=line.get_color())
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("members in the pool  (star = full 14-member pool)")
    ax.set_ylabel("% change vs best single model")
    ax.set_title("Negative is better. Faint dots are the five individual draws.", fontsize=9)
    ax.legend(fontsize=8, ncol=5, frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def plot_featstack_maps(npz_path: Path, out_png: Path, min_support: int = 200,
                        two_panel: bool = False, winner_only: bool = False):
    """Three stacked (variable x hour) maps from a featstack run — the 'when is which model
    better' figure.

      (1) WINNER map      — which architecture has the lowest MSE in each (variable, hour) cell.
      (2) AMBIGUITY share — ambiguity / mean-individual error: how much of the error is model
                            disagreement, i.e. the ceiling on what ANY combiner can recover.
      (3) RESIDUAL corr   — mean pairwise residual correlation; high = models make the same
                            mistake, so nothing to arbitrate.

    Cells with < min_support observations are greyed out in (1): a winner picked from a handful
    of observations is the §13.6a selection-noise trap, and colouring it would invite exactly the
    over-reading this whole analysis exists to avoid.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    z = np.load(npz_path, allow_pickle=True)
    # `winners` is an argmin over the FIT half, so trusting a cell requires fit-half support;
    # greying by `n_eval` alone marked cells trustworthy that were chosen from a handful of
    # fit observations (7-39 per dataset) and hid supported ones (34 on HiRID). A winner is
    # only readable if it was both CHOSEN on enough data and SCORED on enough.
    win = z["winners"]
    n = np.minimum(z["n"], z["n_eval"])
    feats = [str(f) for f in z["features"]]
    models = [str(m) for m in z["models"]]
    amb, mi, rc = z["ambiguity"], z["mean_individual"], z["resid_corr"]

    # Drop variables with NO trustworthy cell at all, rather than drawing them as a solid grey
    # block. Sorted by density they collect at one end -- 25 of UMM's 80 charted channels -- so a
    # third of the figure carried no information and read as a rendering fault.
    keep = (n >= min_support).any(axis=1)
    idx = np.flatnonzero(keep)
    order = idx[np.argsort(-n.sum(axis=1)[idx])]  # densest variables first
    W, N = win[order], n[order]
    share = np.where(mi[order] > 0, amb[order] / np.clip(mi[order], 1e-9, None), np.nan)
    R = rc[order]
    labels = [feats[i] for i in order]

    # Legend by architecture FAMILY: run names carry capacities and hashes that mean nothing to a
    # reader, and two runs of one family would otherwise appear as two legend entries.
    def _fam(nm):
        s = nm.lower()
        for k, v in (("baseline_null", "Null"), ("persist", "LOCF"), ("dense_autoregressive", "DenseAR"),
                     ("hpsf", "DenseAR"), ("ladderms", "DenseAR"), ("dense_dms", "Dense DMS"),
                     ("dlinear", "DLinear"), ("hps_gru", "GRU"), ("gru", "GRU"),
                     ("hps_lstm", "LSTM"), ("lstm", "LSTM"), ("mamba", "Mamba"), ("hpsm", "Mamba"),
                     ("patchtst", "PatchTST"), ("informer", "Informer"), ("autoformer", "Autoformer"),
                     ("strats", "STraTS"), ("tirex", "TiRex-2"), ("moirai", "Moirai")):
            if k in s:
                return v
        return nm[:14]
    fam_names = [_fam(m) for m in models]
    used = sorted(set(W[N >= min_support].ravel().tolist()))
    cmap = plt.get_cmap("tab20", max(len(models), 20))
    Wm = np.where(N >= min_support, W, -1).astype(float)

    # `two_panel` drops the residual-correlation panel. It is not a taste call: on UMM the
    # ambiguity share and 1 - residual correlation correlate at r = 0.93 over the supported cells,
    # so the third panel restates the second at the cost of a third of a full-width figure, and its
    # range (0.82-0.87) is already given as text in the paper.
    # `winner_only` keeps just the winner map, for a slide where three stacked panels
    # render too small to read. squeeze=False so axes[] indexing holds for one panel too.
    n_panels = 1 if winner_only else (2 if two_panel else 3)
    _h = 4.2 if winner_only else (9 if two_panel else 13)
    fig, axes = plt.subplots(n_panels, 1, squeeze=False,
                             figsize=(max(10, len(order) * 0.22), _h))
    axes = axes[:, 0]
    cm = mcolors.ListedColormap(["#dddddd"] + [cmap(i) for i in range(len(models))])
    axes[0].imshow(Wm.T + 1, aspect="auto", cmap=cm, vmin=0, vmax=len(models))
    n_fam = len({fam_names[i] for i in used})
    # State the measured concentration rather than asserting "none owns the panel", which nothing
    # in this function computes.
    cnt = np.bincount(Wm[N >= min_support].astype(int).ravel(), minlength=len(models))
    top = 100.0 * cnt.max() / max(cnt.sum(), 1)
    axes[0].set_title(f"Winning architecture per (variable, hour): {n_fam} architectures win "
                      f"somewhere; the most successful takes {top:.0f}% of cells"
                      f"   [grey = <{min_support} observations]", fontsize=11)
    seen, handles, labs = set(), [], []
    for i in used:                                  # one legend entry per family, not per run
        if fam_names[i] in seen:
            continue
        seen.add(fam_names[i])
        handles.append(plt.Rectangle((0, 0), 1, 1, color=cmap(i)))
        labs.append(fam_names[i])
    # Only the bottom panel carries variable names: repeating 55 rotated labels three times wastes
    # a third of the figure, and the legend previously landed on top of the first copy.
    axes[0].legend(handles, labs, fontsize=7, ncol=7,
                   loc="lower center", bbox_to_anchor=(0.5, -0.30), frameon=False)

    # Scale the ambiguity panel to the DATA, not to [0,1]. The share spans 0.07-0.75 with median
    # 0.21, so on a 0-1 map 95 % of cells sat in the bottom 43 % of the colourmap and the panel read
    # as uniformly dark -- the structure it exists to show was invisible.
    lo, hi = np.nanpercentile(share, [2, 98])
    if winner_only:
        for ax in axes:
            ax.set_yticks(range(0, Wm.shape[1], 4)); ax.set_ylabel("forecast hour")
            ax.set_xticks(range(len(order))); ax.set_xticklabels(order, rotation=90, fontsize=5)
        fig.tight_layout()
        fig.savefig(out_png, dpi=140, bbox_inches="tight")
        fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight")
        plt.close(fig)
        return
    im1 = axes[1].imshow(share.T, aspect="auto", cmap="viridis", vmin=float(lo), vmax=float(hi))
    axes[1].set_title(f"Ambiguity share = diversity / mean individual error  "
                      f"(ceiling on any combiner; scale {lo:.2f}-{hi:.2f})")
    fig.colorbar(im1, ax=axes[1], fraction=0.015, pad=0.006)

    if not two_panel:
        im2 = axes[2].imshow(R.T, aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=1.0)
        axes[2].set_title("Mean pairwise residual correlation (1.0 = same mistake, nothing to gain)")
        fig.colorbar(im2, ax=axes[2], fraction=0.015, pad=0.006)

    for ax in axes:
        ax.set_ylabel("forecast hour")
        ax.set_xticks(range(len(labels)))
        # every panel keeps its channel names: the panels are read individually, and without them
        # the winner map has no statement.
        ax.set_xticklabels(labels, rotation=90, fontsize=5)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)
    return dict(n_vars=len(order), models_used=[models[i] for i in used])


def featstack_consistency(npz_path: Path, min_support: int = 200):
    """How CONSISTENT is the per-variable winner? A routing rule is only meaningful if the same
    model keeps winning a variable across forecast hours; if the winner flips hour to hour, the
    'winner' is noise and per-variable routing is fitting it."""
    z = np.load(npz_path, allow_pickle=True)
    # `winners` is an argmin over the FIT half, so trusting a cell requires fit-half support;
    # greying by `n_eval` alone marked cells trustworthy that were chosen from a handful of
    # fit observations (7-39 per dataset) and hid supported ones (34 on HiRID). A winner is
    # only readable if it was both CHOSEN on enough data and SCORED on enough.
    win = z["winners"]
    n = np.minimum(z["n"], z["n_eval"])
    models = [str(m) for m in z["models"]]
    ok = n >= min_support
    rows = []
    for f in range(win.shape[0]):
        w = win[f][ok[f]]
        if len(w) < 4:
            continue
        vals, cnt = np.unique(w, return_counts=True)
        rows.append(dict(feature=str(z["features"][f]), modal=models[vals[cnt.argmax()]],
                         modal_share=cnt.max() / len(w), n_distinct=len(vals)))
    df = pd.DataFrame(rows)
    return df


def plot_horizon_gain(tables: Path, out_png: Path):
    """Combination gain against the best single model, by horizon step.

    A figure rather than the 24-row table it replaces: the claim is about a TREND -- the gain decays
    as members converge on the same drift -- and a trend is what a table communicates worst.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LBL = {"umm": COHORT, "hirid": "HiRID", "mimic_iii": "MIMIC-III",
           "mimic_iv": "MIMIC-IV", "eicu": "eICU"}
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    for ds, lab in LBL.items():
        f = tables / f"final_byhorizon_{ds}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d = d[d["combiner"] == "simplex_shrunk500_var"].sort_values("horizon")
        ax.plot(d["horizon"], d["delta_pct"], marker="o", ms=3, lw=1.4, label=lab)
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("forecast hour"); ax.set_ylabel("gain vs best single model (%)")
    ax.set_title("Combination gain decays with horizon", fontsize=11)
    ax.legend(fontsize=8, ncol=3); ax.grid(alpha=0.3); ax.set_xticks([1, 4, 8, 12, 16, 20, 24])
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)


def plot_paired_contrasts(tables: Path, out_png: Path):
    """Forest plot of the paired combiner-vs-combiner contrasts.

    Intervals come from a percentile bootstrap over stays; significance from a paired approximate
    randomization test (Noreen). An interval against the best single model says each combiner beats
    one model. It does NOT say
    the combiners differ from EACH OTHER, which is what the per-variable-vs-global and the
    profile claims assert. These are paired: same stay draws, same eligible channels, so the
    contrast is exact rather than two separate intervals compared by eye.

    A figure rather than a 30-row table because the reading is positional -- which intervals clear
    zero and which straddle it -- and that is what a table communicates worst.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    LBL = {"umm": COHORT, "hirid": "HiRID", "mimic_iii": "MIMIC-III",
           "mimic_iv": "MIMIC-IV", "eicu": "eICU"}
    ROWS = [(("simplex_shrunk500_var", "rank_harmonic_top5"), r"$\bf{ENSEMBLE}$ vs harm-5"),
            (("simplex_shrunk500_var", "mean_top5_var"),      r"$\bf{ENSEMBLE}$ vs equal-5"),
            (("simplex_var", "simplex_global"),               "simplex vs simplex-global"),
            (("simplex_shrunk500_var", "simplex_global"),     r"$\bf{ENSEMBLE}$ vs simplex-global"),
            (("rank_harmonic_top5", "mean_top5_var"),         "harm-5 vs equal-5"),
            (("rank_harmonic_top3", "mean_top3_var"),         "harm-3 vs equal-3")]
    frames = [pd.read_csv(f) for f in sorted(tables.glob("final_paired_*.csv"))]
    if not frames:
        return
    D = pd.concat(frames)
    cols = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    order = list(LBL)
    off = np.linspace(0.30, -0.30, len(order))
    for r, ((a, b), lab) in enumerate(ROWS):
        g = D[(D.a == a) & (D.b == b)].set_index("dataset")
        for i, ds in enumerate(order):
            if ds not in g.index:
                continue
            row = g.loc[ds]
            y = -r + off[i]
            sig = row.p_value < 0.05
            ax.plot([row.ci_lo, row.ci_hi], [y, y], color=cols(i), lw=1.5,
                    solid_capstyle="butt", alpha=0.95 if sig else 0.45)
            ax.plot(row.delta_pct, y, "o" if sig else "o", ms=5 if sig else 4.2,
                    color=cols(i), mfc=cols(i) if sig else "white",
                    mew=1.2, zorder=3,
                    label=LBL[ds] if r == 0 else None)
    ax.axvline(0, color="0.35", lw=1.0, ls="--")
    ax.set_yticks([-r for r in range(len(ROWS))])
    ax.set_yticklabels([lab for _, lab in ROWS], fontsize=9)
    ax.set_ylim(-len(ROWS) + 0.45, 0.55)
    ax.set_xlabel("difference in MSE-per-var (%)  |  negative favors the first method",
                  fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(fontsize=8, ncol=5, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              frameon=False, handletextpad=0.3, columnspacing=1.1)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    fig.savefig(str(out_png).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

def plot_channel_coverage(tables: Path, out_png: Path, panels: int = 1):
    """Per-channel stay coverage as a heatmap instead of a 94-row table.

    The table is correct but costs most of an appendix page and is read as a lookup rather than a
    pattern. The pattern is the point: which channels every ICU records, and which are local
    practice.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f = tables / "channel_stay_coverage.csv"
    if not f.exists():
        return
    C = pd.read_csv(f, index_col=0)
    order = [c for c in ["umm", "hirid", "mimic_iii", "mimic_iv", "eicu"] if c in C.columns]
    C = C[order]
    C = C.loc[C.mean(axis=1).sort_values(ascending=False).index]
    xl = [COHORT, "HiRID", "M-III", "M-IV", "eICU"][:C.shape[1]]
    if panels > 1:
        # Page-wide variant: the ranked channel list split into side-by-side panels, so each row
        # gets `panels` times the height and the channel names can be read at print size.
        n = -(-len(C) // panels)
        fig, axes = plt.subplots(1, panels, figsize=(7.0, 8.4))
        fig.subplots_adjust(wspace=0.75)      # room for the right panel's channel names
        for k, ax in enumerate(axes):
            part = C.iloc[k * n:(k + 1) * n]
            im = ax.imshow(part.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
            ax.set_xticks(range(C.shape[1])); ax.set_xticklabels(xl, fontsize=8)
            ax.xaxis.tick_top()
            ax.set_yticks(range(len(part))); ax.set_yticklabels(part.index, fontsize=6.5)
            ax.tick_params(length=0)
        cb = fig.colorbar(im, ax=list(axes), fraction=0.03, pad=0.02)
        cb.set_label("% of stays recording the channel", fontsize=8); cb.ax.tick_params(labelsize=7)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=160, bbox_inches="tight")
        fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)
        return
    fig, ax = plt.subplots(figsize=(4.4, 12.5))
    im = ax.imshow(C.values, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    ax.set_xticks(range(C.shape[1]))
    ax.set_xticklabels([COHORT, "HiRID", "M-III", "M-IV", "eICU"][:C.shape[1]], rotation=45,
                       ha="right", fontsize=8)
    ax.set_yticks(range(len(C))); ax.set_yticklabels(C.index, fontsize=5)
    ax.set_title("% of stays recording each channel", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout(); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    # PDF alongside PNG: LaTeX prefers vector, and the PNG stays for quick viewing.
    fig.savefig(str(out_png).replace('.png', '.pdf'), bbox_inches="tight"); plt.close(fig)



# ---------------------------------------------------------------------------
# Per-variable winners: is the best single model predictable from the channel?
# ---------------------------------------------------------------------------

# Pool-member run names carry their family as a prefix/substring, with no `model` field
# beside them in the .npz (the featstack pool is built from run names alone).
_POOL_FAMILY = (
    ("baseline_null",   "null"),        ("baseline_persistence", "LOCF"),
    ("autoformer",      "autoformer"),  ("dense_dms",  "dense-DMS"),
    ("dlinear",         "DLinear"),     ("hpsf",       "denseAR"),
    ("hps_gru",         "GRU"),         ("hps_lstm",   "LSTM"),
    ("informer",        "informer"),    ("mamba",      "mamba"),
    ("strats",          "STraTS"),      ("patchtst",   "patchTST"),
    ("tirex",           "TiRex"),       ("moirai",     "Moirai"),
    ("chronos2",        "Chronos-2"),   ("chronos",    "Chronos"),
    ("mira",            "MIRA"),
    # The pool also carries pre-hpsweep runs under their own naming: hpsm_* is the
    # mamba sweep, ladderms_* / dense_autoregressive_* are denseAR, and the bare
    # gru_*/lstm_* multi-seed runs predate the hps_ prefix. Order matters: the
    # longer keys above must be tested first.
    ("hpsm",            "mamba"),
    ("ladderms",        "denseAR"),     ("dense_autoregressive", "denseAR"),
    ("gru",             "GRU"),         ("lstm",       "LSTM"),
)


def family_of_run(run: str) -> str:
    r = run.lower()
    for key, fam in _POOL_FAMILY:
        if key in r:
            return fam
    return "other"


def pervar_winners(eval_dir: Path, out_csv: Path, out_png: Path | None = None,
                   min_obs: float = 0.01) -> None:
    """Which single model wins each variable, and does that track sparsity / type / N?

    `featstack_<ds>_val_winners.npz` carries `per_var` (94 variables x 14 pool members),
    `features` and `models`, all on VALIDATION -- the split the stacker is allowed to see.
    argmin over the pool gives the best single model per (dataset, variable), which is
    exactly what the `select_var` combiner would pick. Joined against:
      - observation rate per (channel, dataset) from channel_coverage_by_dataset.csv
      - the channel's `type` (vital / lab / medication / fluid / ventilation / ...) from
        configs/features.yaml
      - the dataset's training-window count
    so the question "do sparse channels, or a whole channel type, prefer one architecture?"
    can be answered from one table rather than by eye.

    Writes a long table (one row per dataset x variable) and, optionally, a figure of the
    winner share per channel type and per sparsity quintile.
    """
    import matplotlib.pyplot as plt
    import yaml
    cov = pd.read_csv(eval_dir / "channel_coverage_by_dataset.csv").set_index("channel")
    cat = yaml.safe_load(open(ROOT / "configs" / "features.yaml"))
    feats = cat.get("features", cat)
    def _type(name):
        e = feats.get(name) or {}
        return (e.get("type") or "unknown") if isinstance(e, dict) else "unknown"

    rows = []
    for npz in sorted(eval_dir.glob("featstack_*_val_winners.npz")):
        ds = npz.name[len("featstack_"):-len("_val_winners.npz")]
        z = np.load(npz, allow_pickle=True)
        pv = np.asarray(z["per_var"], dtype=float)            # (F, M) val MSE
        names = [str(x) for x in z["features"]]
        models = [str(x) for x in z["models"]]
        fam = [family_of_run(m) for m in models]
        for i, v in enumerate(names):
            col = pv[i]
            if not np.isfinite(col).any():
                continue
            j = int(np.nanargmin(col))
            order = np.argsort(np.where(np.isfinite(col), col, np.inf))
            second = float(col[order[1]]) if len(order) > 1 and np.isfinite(col[order[1]]) else np.nan
            rows.append({"dataset": ds, "feature": v, "type": _type(v),
                         "obs_rate": float(cov[ds].get(v, np.nan)) if ds in cov.columns else np.nan,
                         "winner": models[j], "winner_family": fam[j],
                         "mse_winner": float(col[j]), "mse_second": second,
                         "margin_pct": (100.0 * (second - col[j]) / second) if second == second and second else np.nan,
                         "mse_pool_median": float(np.nanmedian(col)),
                         "n_pool": int(np.isfinite(col).sum())})
    df = pd.DataFrame(rows)
    if df.empty:
        print("  pervar: no featstack_*_val_winners.npz found"); return
    # An all-masked or near-empty channel has no signal to fit, so every pool member scores
    # the same degenerate value and argmin returns whichever column happens to come first.
    # Without this filter the result is dominated by that artefact: autoformer "won" 64 % of
    # the sparsest quintile, and 61 of those 95 rows had obs_rate == 0.00000 with an
    # undefined margin over the runner-up. At min_obs = 1 % autoformer's share across every
    # quintile drops to 0-1 %. Keep the dropped rows in the CSV (flagged) but exclude them
    # from every share printed below.
    df["has_signal"] = df["obs_rate"] > min_obs
    n_drop = int((~df["has_signal"]).sum())
    df["sparsity_quintile"] = (df.groupby("dataset")["obs_rate"]
                               .transform(lambda s: pd.qcut(s.rank(method="first"), 5,
                                                            labels=[1, 2, 3, 4, 5])))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  pervar winners: {len(df)} rows -> {out_csv.name}")
    print(f"  dropped {n_drop} (dataset, channel) rows with obs_rate <= {min_obs:.0%} "
          f"(no signal; the winner there is argmin over ties), {len(df)-n_drop} kept")
    df = df[df["has_signal"]].copy()
    df["sparsity_quintile"] = pd.qcut(df.groupby("dataset")["obs_rate"]
                                      .transform(lambda s: s.rank(pct=True, method="first")),
                                      5, labels=[1, 2, 3, 4, 5])

    # (1) does ONE family dominate, or is the winner genuinely per-variable?
    print("\n  winner family share per dataset (% of 94 channels):")
    sh = (df.pivot_table(index="winner_family", columns="dataset", values="feature",
                         aggfunc="count").fillna(0) / df.groupby("dataset")["feature"].count() * 100)
    print(sh.round(1).to_string())

    # (2) the question itself: is the winner a function of sparsity, or of channel type?
    print("\n  winner family share by sparsity quintile (1 = sparsest), pooled over datasets:")
    q = (df.pivot_table(index="winner_family", columns="sparsity_quintile", values="feature",
                        aggfunc="count", observed=False).fillna(0))
    print((q / q.sum() * 100).round(1).to_string())
    print("\n  winner family share by channel type, pooled over datasets:")
    t = (df.pivot_table(index="winner_family", columns="type", values="feature",
                        aggfunc="count").fillna(0))
    print((t / t.sum() * 100).round(1).to_string())

    # (3) how much is actually at stake: if the margin over the runner-up is ~0, the
    # "winner" is noise and a single model would lose nothing.
    print("\n  margin of the winner over the runner-up (%% of runner-up MSE):")
    print(df.groupby("dataset")["margin_pct"].describe()[["mean", "50%", "max"]].round(2).to_string())

    # (4) agreement ACROSS datasets: a channel whose winner is the same everywhere is
    # evidence for a channel property; a channel that changes winner per database is
    # evidence for a dataset property.
    agree = (df.groupby("feature")["winner_family"]
               .agg(lambda s: s.value_counts().iloc[0] / len(s)))
    print(f"\n  cross-database winner agreement: mean {agree.mean():.2f}, "
          f"channels with the same winner in all 5: {(agree >= 0.999).sum()} of {len(agree)}")

    if out_png is not None:
        fams = sorted(df.winner_family.dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
        for ax, (key, lab) in zip(axes, [("sparsity_quintile", "sparsity quintile (1 = sparsest)"),
                                         ("type", "channel type")]):
            pt = (df.pivot_table(index=key, columns="winner_family", values="feature",
                                 aggfunc="count", observed=False).fillna(0))
            pt = pt.reindex(columns=fams, fill_value=0)
            (pt.div(pt.sum(axis=1), axis=0) * 100).plot.bar(stacked=True, ax=ax, width=.8,
                                                            legend=(ax is axes[-1]))
            ax.set_xlabel(lab); ax.set_ylabel("% of channels won"); ax.set_ylim(0, 100)
            ax.tick_params(axis="x", rotation=30)
        if axes[-1].get_legend():
            axes[-1].legend(fontsize=7, ncol=2, loc="upper left", bbox_to_anchor=(1.01, 1.0))
        fig.suptitle("Best single model per variable — by sparsity and by channel type "
                     "(validation, 14-member pool, 5 databases)", fontsize=10)
        fig.tight_layout()
        out_png.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=170, bbox_inches="tight"); plt.close(fig)
        print(f"  wrote {out_png.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints-dir", type=Path, default=ROOT / "checkpoints")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "docs")
    ap.add_argument("--eval-dir", type=Path, default=ROOT / "docs" / "tables",
                    help="dir with FM/precomputed eval_*.json (chronos/chronos2/moirai/mira)")
    ap.add_argument("--no-fm", action="store_true", help="exclude the FM eval JSONs")
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER,
                    help="model buckets to include (default: all)")
    ap.add_argument("--feature-model", default="denseAR",
                    help="model bucket for the per-feature figure")
    ap.add_argument("--heatmap-max-features", type=int, default=None,
                    help="limit the architecture x feature heatmap to the N hardest variables")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--ranked-figure", nargs=2, type=int, metavar=("HARDEST", "EASIEST"),
                    help="only redraw the ranked model x variable figure from the saved "
                         "docs/tables/arch_feature_mse.csv (paper used 40 10; poster v2 uses 18 6)")
    ap.add_argument("--ranked-font-scale", type=float, default=1.0)
    ap.add_argument("--ranked-height-scale", type=float, default=1.0)
    ap.add_argument("--ranked-name", default=None,
                    help="output name in docs/figures/ (default arch_feature_ranked_topH-E.png)")
    ap.add_argument("--map-min-support", type=int, default=200,
                    help="grey out (variable, hour) cells with fewer observations than this")
    ap.add_argument("--pervar-min-obs", type=float, default=0.01,
                    help="drop (dataset, channel) pairs below this observation rate from the "
                         "per-variable winner shares: with no signal every pool member ties "
                         "and the winner is argmin over ties (default 0.01)")
    ap.add_argument("--pervar-winners", action="store_true",
                    help="best single model per (dataset, variable) vs sparsity / channel type "
                         "/ database size, from featstack_*_val_winners.npz")
    ap.add_argument("--coverage-figure", type=int, metavar="PANELS", default=None,
                    help="redraw only the per-channel stay-coverage heatmap, split into PANELS "
                         "side-by-side panels, from channel_stay_coverage.csv")
    args = ap.parse_args()
    if args.coverage_figure:
        out = args.out_dir / "figures" / (f"channel_coverage_{COHORT}"
                                          + (f"_{args.coverage_figure}col" if args.coverage_figure > 1 else "") + ".png")
        plot_channel_coverage(args.eval_dir, out, panels=args.coverage_figure)
        print(f"wrote {out}")
        return
    if args.pervar_winners:
        pervar_winners(args.eval_dir, args.eval_dir / "pervar_winners.csv",
                       ROOT / "docs" / "figures" / "pervar_winners.png",
                       min_obs=args.pervar_min_obs)
        return
    if args.ranked_figure:
        # Redraw only this figure from the stored table, so the paper and poster variants are
        # reproducible without re-running the whole analysis.
        hard, easy = args.ranked_figure
        # keep_default_na=False: the model called "null" must not be parsed as a missing value
        abs_df = pd.read_csv(args.eval_dir / "arch_feature_mse.csv", index_col=0,
                             keep_default_na=False, na_values=[""])
        name = args.ranked_name or f"arch_feature_ranked_top{hard}-{easy}.png"
        plot_arch_feature_ranked(abs_df, args.out_dir / "figures" / name, max_features=hard,
                                 easiest=easy, pool_only=True, font_scale=args.ranked_font_scale,
                                 height_scale=args.ranked_height_scale, model_font_scale=1.0)
        print(f"wrote {args.out_dir / 'figures' / name}")
        return

    best = scan(args.checkpoints_dir)
    if not args.no_fm:
        best.update(scan_fm(args.eval_dir))   # FM buckets are disjoint from checkpoint buckets
    present = sorted({mb for (_, mb) in best})
    models = [m for m in args.models if m in present]
    print(f"Found runs for buckets: {present}")
    print(f"Reporting models: {models}")

    tdir = args.out_dir / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    htab = build_horizon_table(best, models)
    htab.to_csv(tdir / "horizon_mse.csv", index=False)
    long_df, ranked = build_feature_tables(best, models)
    long_df.to_csv(tdir / "feature_mse.csv", index=False)
    ranked.to_csv(tdir / "feature_mse_ranked.csv", index=False)
    common_feature_analysis(long_df, args.out_dir / "figures" / "cross" / "coverage_heatmap_train.csv", tdir)
    # SOFA-relevant feature ranking (folded in from the retired analyze_feature_performance.py;
    # per-model here rather than cross-run, which is the richer view)
    SOFA_VARS = {"pao2", "fio2", "pf_ratio", "mechanical_ventilation", "intubated", "platelets",
                 "bilirubin_total", "creatinine", "urine_output", "map_invasive", "mbp_invasive",
                 "norepinephrine", "epinephrine", "dopamine", "dobutamine",
                 "gcs_eye", "gcs_motor", "gcs_verbal", "gcs_total"}
    if not ranked.empty:
        ranked[ranked["feature"].isin(SOFA_VARS)].to_csv(tdir / "feature_mse_sofa_vars.csv", index=False)
    print(f"Wrote tables → {tdir}/horizon_mse.csv, feature_mse.csv, feature_mse_ranked.csv, feature_mse_sofa_vars.csv")

    per_feature_ensemble(long_df, args.checkpoints_dir, tdir, SOFA_VARS)

    # architecture x feature heatmap (the "do models differ per variable?" view, §13.6/§13.6a)
    abs_df, rel_df = build_arch_feature_matrix(long_df, models)
    if not abs_df.empty:
        abs_df.round(4).to_csv(tdir / "arch_feature_mse.csv")
        rel_df.round(3).to_csv(tdir / "arch_feature_mse_relative.csv")
        spread = rel_df.max(axis=0).sort_values(ascending=False)   # worst/best ratio per variable
        print(f"\nArchitecture x feature: {abs_df.shape[0]} architectures x {abs_df.shape[1]} "
              f"variables (cells covered by every architecture)")
        print(f"  median worst/best spread across variables: {spread.median():.2f}x")
        print("  most architecture-sensitive variables: "
              + ", ".join(f"{f} ({v:.2f}x)" for f, v in spread.head(6).items()))
        print("  least architecture-sensitive: "
              + ", ".join(f"{f} ({v:.2f}x)" for f, v in spread.tail(4).items()))
        # Caveat worth printing, not just knowing: with null/LOCF in the matrix a big spread can
        # mean "a trivial baseline fails here" rather than "architectures differ here". The
        # learned-only spread is the one that speaks to architecture choice.
        learned = [m for m in abs_df.index
                   if m not in {"null", "persistence"} | FM_BUCKETS]
        if len(learned) > 1:
            sub = abs_df.loc[learned]
            sp_l = sub.div(sub.min(axis=0), axis=1).max(axis=0)
            print(f"  learned in-domain only ({len(learned)} archs): median {sp_l.median():.2f}x")
            print("  per-variable winners: "
                  + ", ".join(f"{m}x{n}" for m, n in sub.idxmin(axis=0).value_counts().head(5).items()))
        print(f"  wrote {tdir}/arch_feature_mse.csv, arch_feature_mse_relative.csv")

    # stdout summary
    if not htab.empty:
        print("\nHorizon degradation (h1 → h24, ratio):")
        for _, r in htab.iterrows():
            print(f"  {r['dataset']:11s} {r['model']:13s} "
                  f"h1={r['h1']:.3f}  h24={r['h24']:.3f}  ratio={r['ratio_h24_h1']:.2f}x")

    if not args.no_figures:
        try:
            plot_horizon(htab, models, args.out_dir / "figures" / "horizon_degradation.png")
            plot_features(ranked, args.feature_model,
                          args.out_dir / "figures" / f"feature_mse_{args.feature_model}.png")
            plot_arch_feature_heatmap(abs_df, rel_df,
                                      args.out_dir / "figures" / "arch_feature_heatmap.png",
                                      max_features=args.heatmap_max_features)
            plot_arch_feature_ranked(abs_df,
                                     args.out_dir / "figures" / "arch_feature_ranked.png",
                                     max_features=args.heatmap_max_features)
            plot_arch_feature_colnorm(abs_df,
                                      args.out_dir / "figures" / "arch_feature_colnorm.png",
                                      max_features=args.heatmap_max_features)
            # plot_featstack_maps existed but was never called from main, so the published maps
            # were produced ad hoc and covered only 3 of 5 datasets. Drive it from the artefacts
            # so every dataset with a winners.npz gets a current triptych.
            made = []
            for npz in sorted(args.eval_dir.glob("featstack_*_val_winners.npz")):
                ds = npz.name[len("featstack_"):-len("_val_winners.npz")]
                out = args.out_dir / "figures" / f"featstack_{ds}_maps.png"
                try:
                    plot_featstack_maps(npz, out, min_support=args.map_min_support)
                    # Second variant under its own name so both can be compared before one is
                    # chosen; neither overwrites the other.
                    plot_featstack_maps(
                        npz, args.out_dir / "figures" / f"featstack_{ds}_maps2p.png",
                        min_support=args.map_min_support, two_panel=True)
                    made.append(ds)
                except Exception as e:                      # one bad dataset must not kill the rest
                    print(f"[featstack maps {ds} skipped: {e}]")
            if made:
                print(f"Wrote featstack maps for: {', '.join(made)}")
            print(f"Wrote figures → {args.out_dir}/figures/")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[figures skipped: {e}]")


_STATIC_FEATS = {"age", "sex", "height", "los_hours", "mortality_icu", "weight_admission", "bmi"}


def common_feature_analysis(long_df, coverage_csv: Path, tdir: Path, thresh: float = 0.10):
    """Restrict the per-variable test MSE to the features actually charted (coverage ≥ thresh)
    in EVERY dataset → a cross-dataset-comparable model ladder (the full-94 aggregate is
    confounded by each dataset's differing feature mix + 0.0 placeholders for unavailable
    features). Plus a per-(variable, dataset) predictability matrix (the --feature-model bucket)
    flagging variables much harder in one dataset than the rest — a late data-selection /
    notation-quality signal (a well-charted variable that is unpredictable only in dataset X
    points at X's charting of it, not the model). Reads the coverage matrix (features × datasets,
    docs/figures/cross/coverage_heatmap_train.csv)."""
    import numpy as np
    import pandas as pd
    if not coverage_csv.exists():
        print(f"[common-feature analysis skipped: {coverage_csv} not found]")
        return
    cov = pd.read_csv(coverage_csv, index_col=0)
    dsc = [d for d in DATASETS if d in cov.columns]
    fcfeats = set(long_df["feature"].dropna())
    common = sorted(f for f in cov.index
                    if f in fcfeats and f not in _STATIC_FEATS
                    and bool((cov.loc[f, dsc] >= thresh).all()))
    if not common:
        print("[common-feature analysis: no common features]")
        return
    # (a) cross-dataset-comparable ladder: per (model, dataset) mean over the common features
    sub = long_df[long_df["feature"].isin(common)]
    ladder = (sub.pivot_table(index="model", columns="dataset", values="mse", aggfunc="mean")
              .reindex(columns=dsc))
    ladder["mean"] = ladder.mean(axis=1)
    ladder = ladder.sort_values("mean")
    ladder.round(4).to_csv(tdir / "common_feature_ladder.csv")
    # (b) predictability matrix (denseAR by default): common feature × dataset + worst + spread
    fmname = "denseAR" if "denseAR" in set(long_df["model"]) else ladder.index[0]
    mat = (long_df[long_df["model"] == fmname]
           .pivot_table(index="feature", columns="dataset", values="mse")
           .reindex(index=common, columns=dsc))
    mat["worst_dataset"] = mat[dsc].idxmax(axis=1)
    mat["spread"] = mat[dsc].max(axis=1) - mat[dsc].min(axis=1)
    mat.sort_values("spread", ascending=False).round(4).to_csv(tdir / "common_feature_predictability.csv")
    print(f"\nCommon-feature set (coverage ≥ {thresh:.0%} in all {len(dsc)} datasets): {len(common)} forecast vars")
    if fmname in ladder.index:
        r = ladder.loc[fmname]
        sp = r[dsc].max() - r[dsc].min()
        print(f"  {fmname} per-dataset mean over common feats: "
              + ", ".join(f"{d} {r[d]:.3f}" for d in dsc) + f"  (spread {sp:.3f})")
    print(f"  → {tdir}/common_feature_ladder.csv, common_feature_predictability.csv")


def gather_bucket_runs(ckpt_dir: Path, want_bucket: str) -> dict:
    """All runs' val feat_mse for one model bucket, keyed dataset -> list[{feature: mse}].
    Unlike scan() (which keeps only the best run per bucket) this keeps EVERY run so the
    per-feature run-to-run (seed/HP) spread can be measured."""
    import json
    out: dict[str, list] = {}
    for jf in ckpt_dir.glob("*/final_metrics.json"):
        try:
            j = json.loads(jf.read_text())
        except Exception:
            continue
        combo = j.get("combo", {}) or {}
        if bucket(jf.parent.name, combo.get("model", "")) != want_bucket:
            continue
        ds = dataset_of(combo)
        fm = (j.get("val", {}) or {}).get("feat_mse")
        if ds and fm:
            out.setdefault(ds, []).append(fm)
    return out


def per_feature_ensemble(long_df, ckpt_dir: Path, tdir: Path, sofa_vars: set,
                         ref: str = "denseAR", trivial: float = 0.05):
    """Per-variable model-selection ("switching ensemble") ablation, from the val
    per-(dataset, model, feature) MSE in long_df (z-scored units).

    Asks: do different model families predict different variables best, enough that
    routing each output channel to its best model would beat the single best model?
    Reports, per dataset:
      - best single model vs the FULL oracle (best-of-all-models per feature) — the
        OPTIMISTIC upper bound (min over N models on the same val data it is scored on);
      - CONSENSUS routing (pick the winner per feature by cross-dataset *mean*, then apply
        that one fixed rule per dataset) — the honest, generalisable gain (cannot overfit
        per-dataset selection noise);
      - the oracle on the SOFA-relevant subset (does the headroom survive clinically?);
      - the greedy REDUCED oracle (denseAR + best 1/2/3 partners) — how few models capture it;
      - per-feature run-to-run (seed/HP) std vs model-to-model std — how much of the oracle is
        selection-over-noise (min-of-N of noisy draws ≈ 1.5·σ);
      - winner cross-dataset consistency + the robust routing rules (win ≥3/5 datasets AND
        ≥15% better than the reference).
    Writes per_feature_ensemble_{summary,reduced,rules}.csv."""
    import numpy as np
    import pandas as pd

    d = long_df.dropna(subset=["model", "feature", "mse"])
    d = d[d["mse"] > 0]
    DS = [x for x in DATASETS if x in set(d["dataset"])]
    if ref not in set(d["model"]):
        print(f"[per_feature_ensemble skipped: no {ref} runs]")
        return
    # consensus routing rule (feature -> model, by cross-dataset mean MSE)
    mean_fm = d.groupby(["feature", "model"])["mse"].mean()
    consensus = mean_fm.groupby(level=0).idxmin().map(lambda t: t[1])
    # per-feature winner per dataset (for consistency)
    win = d.loc[d.groupby(["dataset", "feature"])["mse"].idxmin()]
    wc = {f: g["model"].value_counts() for f, g in win.groupby("feature")}
    # run-to-run spread of the reference model
    runs = gather_bucket_runs(ckpt_dir, ref)

    summ, reduced = [], []
    for ds in DS:
        s = d[d["dataset"] == ds]
        piv = s.pivot_table(index="feature", columns="model", values="mse")
        per_model = s.groupby("model")["mse"].mean()
        best_model, best = per_model.idxmin(), per_model.min()
        oracle = piv.min(axis=1).mean()
        con_vals = [piv.loc[f, consensus[f]] for f in piv.index
                    if consensus.get(f) in piv.columns and pd.notna(piv.loc[f, consensus[f]])]
        consensus_mse = float(np.mean(con_vals))
        sv_feats = [f for f in piv.index if f in sofa_vars]
        oracle_sofa = float(piv.loc[sv_feats].min(axis=1).mean()) if sv_feats else np.nan
        best_sofa = (s[s["feature"].isin(sofa_vars)].groupby("model")["mse"].mean().min()
                     if sv_feats else np.nan)
        n_trivial = int((piv[ref] < trivial).sum()) if ref in piv.columns else -1
        # run-to-run vs model-to-model spread on real (non-trivial) channels
        rr = runs.get(ds, [])
        if len(rr) >= 2:
            feats = set.intersection(*[set(r) for r in rr])
            seed_std = float(np.mean([np.std([r[f] for r in rr]) for f in feats
                                      if all(r[f] > trivial for r in rr)]))
        else:
            seed_std = np.nan
        model_std = float(s[s["mse"] > trivial].groupby("feature")["mse"].std().mean())
        # greedy reduced oracle: ref + best partner(s)
        if ref in piv.columns:
            chosen, cur, base = [ref], piv[ref].copy(), piv[ref].mean()
            reduced.append({"dataset": ds, "step": ref, "mse": round(float(base), 4), "gain_pct": 0.0})
            for _ in range(3):
                bm, bv = None, cur.mean()
                for m in piv.columns:
                    if m in chosen:
                        continue
                    v = np.minimum(cur, piv[m]).mean()
                    if v < bv:
                        bv, bm = v, m
                if bm is None:
                    break
                chosen.append(bm); cur = np.minimum(cur, piv[bm])
                reduced.append({"dataset": ds, "step": f"+{bm}", "mse": round(float(bv), 4),
                                "gain_pct": round(100 * (1 - bv / base), 1)})
        summ.append({
            "dataset": ds, "best_model": best_model, "best_mse": round(float(best), 4),
            "oracle_mse": round(float(oracle), 4), "oracle_gain_pct": round(100 * (1 - oracle / best), 1),
            "consensus_mse": round(consensus_mse, 4),
            "consensus_gain_pct": round(100 * (1 - consensus_mse / best), 1),
            "best_sofa": round(float(best_sofa), 4) if sv_feats else None,
            "oracle_sofa": round(oracle_sofa, 4) if sv_feats else None,
            "oracle_gain_sofa_pct": round(100 * (1 - oracle_sofa / best_sofa), 1) if sv_feats else None,
            "n_trivial_ch": n_trivial,
            "seed_std": round(seed_std, 4) if not np.isnan(seed_std) else None,
            "model_std": round(model_std, 4),
            "model_over_seed": round(model_std / seed_std, 1) if seed_std and not np.isnan(seed_std) else None,
            "noise_frac_of_oracle": (round(min(1.0, 1.53 * seed_std / (best - oracle)), 2)
                                     if seed_std and not np.isnan(seed_std) and best > oracle else None),
        })
    summ_df = pd.DataFrame(summ)
    summ_df.to_csv(tdir / "per_feature_ensemble_summary.csv", index=False)
    pd.DataFrame(reduced).to_csv(tdir / "per_feature_ensemble_reduced.csv", index=False)

    # robust routing rules: non-ref wins in >=3/5 datasets AND >=15% better on cross-ds mean
    mm = mean_fm.unstack()
    rules = []
    for f in mm.index:
        if ref not in mm.columns or pd.isna(mm.loc[f, ref]) or mm.loc[f, ref] <= 0:
            continue
        alt = mm.loc[f].drop(ref)
        bm, bv = alt.idxmin(), alt.min()
        if pd.isna(bv):
            continue
        vc = wc.get(f)
        if vc is None:
            continue
        gain = 1 - bv / mm.loc[f, ref]
        if gain >= 0.15 and int(vc.iloc[0]) >= 3 and vc.index[0] == bm:
            rules.append({"feature": f, "ref_mse": round(float(mm.loc[f, ref]), 3),
                          "best_model": bm, "best_mse": round(float(bv), 3),
                          "gain_pct": round(100 * gain, 0), "wins_in": f"{int(vc.iloc[0])}/5"})
    pd.DataFrame(rules).to_csv(tdir / "per_feature_ensemble_rules.csv", index=False)

    # stdout summary
    print("\n=== Per-variable switching-ensemble ablation (val z-MSE) ===")
    for _, r in summ_df.iterrows():
        print(f"  {r['dataset']:11s} best={r['best_model']}({r['best_mse']:.3f}) | "
              f"oracle {r['oracle_gain_pct']:+.0f}% | consensus {r['consensus_gain_pct']:+.0f}% | "
              f"SOFA-oracle {r['oracle_gain_sofa_pct'] if r['oracle_gain_sofa_pct'] is not None else float('nan'):+.0f}% | "
              f"model/seed spread {r['model_over_seed']}x | oracle noise-frac {r['noise_frac_of_oracle']}")
    print(f"  robust routing rules (win ≥3/5 & ≥15%): {len(rules)} of {mm.shape[0]} features")
    for r in rules:
        print(f"    {r['feature']:16s} -> {r['best_model']:12s} {r['gain_pct']:.0f}% better, wins {r['wins_in']}")
    print(f"  → {tdir}/per_feature_ensemble_{{summary,reduced,rules}}.csv")


if __name__ == "__main__":
    main()
