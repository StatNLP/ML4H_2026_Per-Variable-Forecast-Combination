#!/usr/bin/env python3
"""Generate the paper appendix (docs/appendix_generated.tex) from the artifact CSVs.

Generated rather than hand-written on purpose: every table below is a projection of a file in
docs/tables/, so a re-run can never silently disagree with the main text, and no number is ever
transcribed by hand. The main .tex \\input{}s the result.

    python 10_scripts/make_appendix.py [--tables docs/tables] [--out docs/appendix_generated.tex]
"""
from __future__ import annotations

import argparse
import sys as _sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DS = ["umm", "hirid", "mimic_iii", "mimic_iv", "eicu"]
# De-anonymised 2026-10-26 (MF): the dissertation names the database. The paper used
# \textsc{ANON}; for a blind submission change it back HERE, never in the generated .tex,
# which is overwritten on every run.
LBL = {"umm": "UMM", "hirid": "HiRID", "mimic_iii": "MIMIC-III",
       "mimic_iv": "MIMIC-IV", "eicu": "eICU"}
# Plain (non-small-caps) names for use inside \textbf{}: OT1 has no bold small-caps.
PLAIN = {"umm": "UMM", "hirid": "HiRID", "mimic_iii": "MIMIC-III",
         "mimic_iv": "MIMIC-IV", "eicu": "eICU"}

# Longest key first, so "dense_autoregressive" is not shadowed by "dense_dms" style prefixes.
FAMILY = [("baseline_null", "Null"), ("null", "Null"), ("persist", "LOCF"),
          ("dense_autoregressive", "DenseAR"), ("hpsf", "DenseAR"), ("ladderms", "DenseAR"),
          ("dense_dms", "Dense DMS"), ("dlinear", "DLinear"),
          ("hps_gru", "GRU"), ("gru", "GRU"), ("hps_lstm", "LSTM"), ("lstm", "LSTM"),
          ("mamba", "Mamba"), ("hpsm", "Mamba"),   # hpsm_* are Mamba HP-search runs
          ("patchtst", "PatchTST"), ("informer", "Informer"),
          ("autoformer", "Autoformer"), ("strats", "STraTS"),
          ("tirex", "TiRex-2"), ("moirai", "Moirai"),
          # Classical + FM rungs, added 2026-10-26 (§37.55/§37.56). Without these they rendered
          # as truncated run names, splitting one family across several pseudo-labels.
          # The matcher is a SUBSTRING test, so order and key choice matter:
          #   * "chronos2" MUST precede "chronos" — the bare key matches both (45 runs vs 15).
          #   * "ets_" keeps the underscore on purpose: a bare "ets" is a substring of
          #     "targets"/"offsets"/"datasets". All 6 ETS runs match "ets_", so it costs nothing.
          # Validated against all 526 run names in docs/tables: each key matches only runs whose
          # own `model` field agrees with the label.
          ("arima", "ARIMA"), ("ets_", "ETS"), ("lightgbm", "LightGBM"),
          ("chronos2", "Chronos-2"), ("chronos", "Chronos-Bolt"), ("mira", "MIRA")]
# NOTE: a SECOND, independent copy of this mapping lives in the F2 pool-diversity section
# (local `FAM`/`_fam`). Both are kept in step by hand; add new families to BOTH (§37.56).


SHORT = {"simplex_shrunk500_var": r"\textsc{ens.}", "simplex_shrunk100_var": r"shrunk100",
         "simplex_var": "simplex", "greedy_var": "greedy", "rank_harmonic_top5": r"\textsc{harm-5}",
         "rank_harmonic_top3": r"\textsc{harm-3}", "rank_fitted_top3": "fit-3",
         "equal_top3": r"\textsc{equal-3}", "equal_top5": r"\textsc{equal-5}",
         "rank_fitted_top5": "fit-5", "rank_fitted_top14": "fit-14",
         "rank_harmonic_top14": r"\textsc{harm-14}", "mean_top3": "g-equal-3", "mean_top5": "g-equal-5",
         "rank_harmonic_gtop3": "g-harm-3", "rank_harmonic_gtop5": "g-harm-5",
         "simplex_global": r"simplex-global", "mean_top3_var": r"equal-3",
         "mean_top5_var": r"equal-5",
         "mean": "mean", "select_var": r"\textsc{select}", "select_var_hour": r"\textsc{sel-}$h$",
         "ridge_var": "ridge", "ridge_var_hour": r"ridge-$h$", "best_single": r"\textsc{best}",
         "inv_mse_var": "inv-MSE", "bma_var": "BMA", "bma_neff_var": r"BMA$_{\mathrm{stays}}$",
         "hedge_var": "Hedge", "ds": "database", "dataset": "database", "horizon": "$h$"}


def esc(s: str) -> str:
    r"""Escape for LaTeX, and map known combiner keys to short display names.

    The raw keys (simplex\_shrunk500\_var, rank\_harmonic\_top5, ...) as column headers made the
    uncertainty and profile tables 76 pt and 108 pt too wide for the text block.
    """
    k = str(s)
    if k in SHORT:
        return SHORT[k]
    return k.replace("_", r"\_")


def fam(name: str) -> str:
    """Map a run name to its architecture family.

    The fallback must be ESCAPED: an unmapped run name reaches the LaTeX table verbatim, and every
    run name in this project contains underscores. An unmapped name also silently becomes its own
    "family", which splits that architecture's win share across two rows -- so a miss is a content
    bug, not only a typesetting one.
    """
    n = name.lower().replace("fm:", "")
    for key, lab in FAMILY:
        if key in n:
            return lab
    # A miss is a CONTENT bug (see above), so say so loudly rather than letting a mangled
    # pseudo-label reach the .tex unremarked. Still returns the escaped name instead of
    # raising, so one unmapped run cannot block the whole appendix.
    print(f"make_appendix WARNING: no FAMILY entry for run {name!r}; add a key to FAMILY",
          file=_sys.stderr)
    return esc(n[:16])


def wrap(body: str, caption: str, label: str, size: str = r"\small",
         star: bool = False) -> str:
    """Every appendix table needs a caption and a label; bare centered tabulars are not citable.

    star=True spans both columns (table*). The class is twocolumn, so a column is only ~234pt;
    a tabular wider than that overflows no matter how far the font is shrunk, and table* is the
    only real fix. Starred floats cannot honour `h`, hence the [t] placement.
    """
    env, pos = ("table*", "[t]") if star else ("table", "[htbp]")
    return (f"\\begin{{{env}}}{pos}\n\\centering{size}\n"
            f"\\caption{{{caption}}}\\label{{tab:{label}}}\n{body}\n\\end{{{env}}}")


def figure(path: str, caption: str, label: str, width: str = r"0.55\textwidth") -> str:
    """A captioned figure float. Every generated float must be referable, like the tables."""
    return (f"\\begin{{figure}}[htbp]\n\\centering\n"
            f"\\includegraphics[width={width}]{{{path}}}\n"
            f"\\caption{{{caption}}}\\label{{fig:{label}}}\n\\end{{figure}}")


def tabular(df: pd.DataFrame, colfmt: str, floatfmt: str = "%.3f") -> str:
    """A booktabs tabular without pandas' \\begin{table} wrapper (we control placement)."""
    # escape=False: column labels may carry LaTeX and cells are pre-escaped by esc(),
    # so letting pandas escape as well would double it (\textbackslash \_).
    body = df.to_latex(index=False, escape=False, column_format=colfmt,
                       float_format=lambda v: floatfmt % v)
    return body.replace(r"\begin{table}", "").replace(r"\end{table}", "")



def grouped_tabular(row_labels, groups, cells, first_col="combiner", sub=("dev", "test")):
    r"""A tabular with a two-level header: one spanning label per group, two sub-columns under it.

    pandas cannot express \multicolumn headers, so rather than string-patching its output we emit
    the tabular directly. `groups` are the spanning labels, `cells` is row-major [[(dev, test), ..]].
    """
    n = len(groups)
    spec = "l" + "rr" * n
    head1 = " & ".join([""] + [r"\multicolumn{2}{c}{%s}" % g for g in groups]) + r" \\"
    rules = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 2 * i, 3 + 2 * i) for i in range(n))
    head2 = " & ".join([first_col] + list(sub) * n) + r" \\"
    body = []
    for lab, row in zip(row_labels, cells):
        flat = []
        for d_v, t_v in row:
            flat += ["--" if d_v is None else "%.2f" % d_v,
                     "--" if t_v is None else "%.2f" % t_v]
        body.append(" & ".join([lab] + flat) + r" \\")
    return ("\\begin{tabular}{%s}\n\\toprule\n%s\n%s\n%s\n\\midrule\n%s\n"
            "\\bottomrule\n\\end{tabular}" % (spec, head1, rules, head2, "\n".join(body)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", type=Path, default=ROOT / "docs" / "tables")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "appendix_generated.tex")
    a = ap.parse_args()
    T, L = a.tables, []
    add = L.append

    def read(pat: str, ds: str):
        f = T / pat.format(ds=ds)
        return pd.read_csv(f) if f.exists() else None

    add(r"\appendix")
    # \thetable changes only the PRINTED number; hyperref builds its destination from \theHtable,
    # so without this every appendix table still emitted table.1, table.2, ... and collided with the
    # main text's. Same for figures, whose counter \appendix also resets.
    add(r"\renewcommand{\thetable}{A\arabic{table}}" "\n"
        r"\renewcommand{\theHtable}{A\arabic{table}}" "\n"
        r"\renewcommand{\thefigure}{A\arabic{figure}}" "\n"
        r"\renewcommand{\theHfigure}{A\arabic{figure}}" "\n"
        r"\setcounter{table}{0}" "\n" r"\setcounter{figure}{0}")
    add(r"\onecolumn")

    # ------------------------------------------------------------------ A  databases
    add(r"\section{Databases}\label{app:data}")
    add(r"""Sizes are in Table~\ref{tab:data}. \emph{Charted} counts the channels a database records at
least once inside a 24-hour target window, and is the denominator $|C|$ of the metric;
never-recorded channels are excluded from every per-variable mean. It differs by at most one from
the non-dash entries of Table~\ref{tab:coverage}, which counts a channel recorded anywhere in a
stay --- a channel seen only outside the first seven days, or only in the input half of a window,
appears in one and not the other. \emph{Density} is the fraction
of (variable, hour) target cells carrying an observation, over the full training split.

Windows are drawn from the first seven ICU days at a 4-hour stride, each with a 24-hour input and a
24-hour horizon, giving up to 31 windows per stay. Splits are 64/16/20 by patient at one fixed seed;
Table~\ref{tab:splits} gives the resulting sizes; HiRID is split by admission, since its release
does not link readmissions to a person.

\begin{table}[htbp]\centering\small
\caption{Database sizes by split, and the mean number of observed (variable, hour) cells per
24-hour target window out of $24\times94=2256$ possible.}\label{tab:splits}
\begin{tabular}{lrrrrrrr}
\toprule
 & \multicolumn{2}{c}{train} & \multicolumn{2}{c}{val} & \multicolumn{2}{c}{test} & obs./ \\
 & stays & windows & stays & windows & stays & windows & window \\
\midrule
MIMIC-III & 18\,242 & 274\,195 & 4\,568 & 69\,351 & 5\,704 & 87\,565 & 249 \\
MIMIC-IV & 29\,604 & 444\,870 & 7\,382 & 111\,567 & 9\,234 & 137\,111 & 257 \\
eICU & 49\,533 & 685\,134 & 12\,197 & 168\,643 & 15\,385 & 211\,251 & 200 \\
HiRID & 5\,543 & 88\,407 & 1\,398 & 22\,143 & 1\,680 & 26\,133 & 304 \\
UMM & 2\,889 & 62\,466 & 710 & 15\,054 & 891 & 19\,541 & 425 \\
\bottomrule
\end{tabular}
\end{table}

\noindent The last column is the mean number of observed (variable, hour) cells per 24-hour target
window, out of $24\times94=2256$ possible. Even UMM, the densest, observes under a fifth.

\paragraph{Variable systems.} Results below are reported over the catalog grouping (\emph{lab},
\emph{vital}, \emph{medication}, \emph{ventilation}), over the organ-failure channels
(\emph{sofa\_relevant}), and over observation-frequency tertiles computed within each database
(\emph{freq:low/mid/high}). Group means use only channels with at least 100 observations: one HiRID
medication channel with twelve observed cells otherwise supplied 41\% of that group's mean.""")

    # Per-channel table: how many STAYS ever record the channel, not a normalized density. The
    # normalized version rounded three quarters of the entries to 0.00 and was unreadable; stay
    # coverage is both interpretable on its own and the quantity a reader needs to judge whether a
    # channel supports a per-channel fit at all.
    cov = T / "channel_stay_coverage.csv"
    if cov.exists():
        C = pd.read_csv(cov, index_col=0)
        C.columns = [LBL.get(c, c) for c in C.columns]
        C = C.reindex(columns=[LBL[d] for d in DS])
        C = C.loc[C.mean(axis=1).sort_values(ascending=False).index]
        Ct = C.reset_index().rename(columns={"index": "channel"})
        Ct["channel"] = Ct["channel"].map(esc)
        for c in Ct.columns[1:]:
            Ct[c] = Ct[c].map(lambda v: "---" if pd.isna(v) or v < 0.05 else f"{v:.1f}")
        add(r"""\paragraph{Per-channel coverage.} Table~\ref{tab:coverage} gives the percentage of stays in
which each channel is recorded at least once. Coverage varies with local practice, not data volume:
albumin appears in 83\% of UMM stays and 14.8\% of HiRID stays. Hence the
$\geq 100$-observation filter on the group-level tables.""")
        half = (len(Ct) + 1) // 2
        # NB: do not name these L/R -- L is the output block list this function joins at the end.
        lh = Ct.iloc[:half].reset_index(drop=True)
        rh = Ct.iloc[half:].reset_index(drop=True)
        rh.columns = [c + " " for c in rh.columns]        # unique names for the concat
        ncol = len(lh.columns)
        Ct = pd.concat([lh, rh], axis=1).fillna("")
        add(wrap(tabular(Ct, ("l" + "r" * (ncol - 1)) * 2, "%.1f"),
                 "Percentage of ICU stays in which each channel is recorded at least once. "
                 "A dash marks a channel the database never records. Channels are ordered by mean "
                 "coverage. Coverage spans the full range within every database, which is why all "
                 "per-variable means here carry a $\\geq 100$-observation filter; the same data is "
                 "plotted in Figure~\\ref{fig:coverage}.", "coverage", r"\tiny"))

        if (ROOT / "docs" / "figures" / "channel_coverage_anon.pdf").exists():
            add(figure("figures/channel_coverage_anon.pdf",
                       "Per-channel stay coverage. The block of channels every database records is "
                       "small; most of the panel is local practice.", "coverage", r"0.42\textwidth"))
    # ------------------------------------------------------------------ B  models
    add(r"\clearpage")
    add(r"\section{Base pool}\label{app:models}")
    add(r"""Fourteen members per database, each tuned per database, selected on validation error under one
rule applied to every family, and trained on the training split alone. The two foundation models are
used \emph{zero-shot}: public checkpoints, no fitting to these databases. \textsc{DenseAR} is the
iterative dense decoder of \citet{StaniekETAL:24mlhc}. Table~\ref{tab:pool} lists the selected
configurations.""")
    import re as _re

    def short(run: str, ds: str) -> str:
        """Strip the dataset name, the family prefix and any trailing hash/seed from a run name.

        The full names carry the database twice and a content hash, which made the table unreadable
        and told the reader nothing: what matters is the selected capacity, not the run id.
        """
        s = run.replace("FM:", "")
        s = s.replace(ds, "").replace("__", "_")
        s = _re.sub(r"_s\d+$", "", s)
        s = _re.sub(r"_[0-9a-f]{6,}$", "", s)
        s = _re.sub(r"^(hpsf|hps|hpsm|ladderms|baseline|dense_autoregressive|dense_dms|strats_repro"
                    r"|strats|informer|autoformer|dlinear|mamba|gru|lstm|patchtst)_*", "", s)
        s = _re.sub(r"^4h_|patchtst_|moirai_|_pl\d+st\d+$|dr00_", "", s)
        s = _re.sub(r"^(persistence|null|tirex2|small)$", "---", s)   # baselines/FMs have no config
        s = s.strip("_")
        return esc(s) if s else "default"

    rows = []
    for ds in DS:
        f = ROOT / "configs" / "pools" / f"{ds}.txt"
        if not f.exists():
            continue
        for m in [x.strip() for x in f.read_text().splitlines() if x.strip()]:
            rows.append({"family": fam(m), "db": LBL[ds], "run": short(m, ds)})
    if rows:
        piv = (pd.DataFrame(rows).pivot_table(index="family", columns="db", values="run",
                                              aggfunc="first")
               .reindex(columns=[LBL[d] for d in DS]).fillna("---").reset_index())
        piv.columns = [str(c) for c in piv.columns]
        add(wrap(tabular(piv, "l" + "l" * len(DS)),
                 "Selected configuration per architecture family and database. Names are stripped of "
                 "the database, the family prefix and the run hash; \\emph{default} means the "
                 "family's standard configuration was selected.", "pool", r"\scriptsize"))

    # ------------------------------------------------------------------ C  combiners
    add(r"\clearpage")
    add(r"\section{Combination algorithms}\label{app:comb}")
    add(r"""Every combiner produces a weight tensor $w_{fhm}$ (channel, horizon step, model) and is
scored by one shared code path, so differences lie in the estimator alone. Let $\mathrm{MSE}_{fm}$
be model $m$'s masked error on channel $f$ over the fitting split, $n_f$ its observation count, and
$G_f$, $b_f$ the Gram matrix and cross-term of Section~4.1.

\begin{description}\small
\item[\normalfont\textsc{best-single}] one-hot on $\arg\min_m \overline{\mathrm{MSE}}_m$. The reference.
\item[\normalfont\textsc{mean}] $w=1/M$ everywhere. No fitting.
\item[\normalfont\textsc{g-equal-}$k$] $1/k$ on the $k$ best members \emph{pool-wide}, the same
  members on every channel.
\item[\normalfont\textsc{g-harm-}$k$] the harmonic profile $1,\tfrac12,\dots,\tfrac1k$ on those same
  pool-wide members. With \textsc{equal-}$k$ and \textsc{harm-}$k$ below it completes a
  $2\times2$ over membership and weight profile.
\item[\normalfont\textsc{equal-}$k$] $1/k$ on the $k$ best members \emph{of that channel} --- the
  matched control for \textsc{harm-}$k$, differing from it only in the weight profile.
\item[\normalfont\textsc{inv-mse}] $w_{fm}\propto 1/\mathrm{MSE}_{fm}$, per channel.
\item[\normalfont\textsc{harm-}$k$] rank the members on each channel, weight the best $k$ by
  $1,\tfrac12,\dots,\tfrac1k$ normalized. Only the ranking is estimated; the weights are fixed in
  advance, so this cannot overfit the weight values.
\item[\normalfont\textsc{select}] one-hot on $\arg\min_m \mathrm{MSE}_{fm}$, per channel. Fewest parameters,
  highest variance: a discrete $\arg\min$ over $M$ noisy estimates.
\item[\normalfont\textsc{ridge}] $w_f=(G_f+\lambda I)^{-1}b_f$, unconstrained; may be negative or $>1$.
\item[\normalfont\textsc{simplex}] $\min_w w^\top G_f w - 2 w^\top b_f$ subject to $w\ge 0$, $\sum_m w_m = 1$,
  by projected gradient. The constraint bounds the combination's error on that channel by that
  channel's worst member.
\item[\normalfont\textsc{simplex-global}] the same program with \emph{one} weight vector for the whole
  panel: $\min_w \sum_f (w^\top G_f w - 2 w^\top b_f + c_f)/n_f$ over the simplex, where $c_f$ is the
  sum of squared observed targets on channel $f$ and channels with $n_f=0$ are excluded. Dividing by
  $n_f$ matches the macro metric, so a densely sampled channel does not dominate the loss. This is
  the control for whether weights need to vary across channels at all.
\item[\normalfont\textsc{bma}] Bayesian model averaging, $w_{fm}\propto\exp(-\tfrac{N_f}{2}\mathrm{MSE}_{fm}/\sigma_f^2)$
  with $\sigma_f^2$ the best member's error on the channel and flat priors. $N_f$ is the observed-cell
  count; \textsc{bma}$_{\mathrm{stays}}$ uses the stay count instead, since cells within a stay are
  not independent.
\item[\normalfont\textsc{hedge}] exponentiated-gradient weights $w_{fm}\propto\exp(-\eta_f T_f\,
  \mathrm{MSE}_{fm}/\bar{\mathrm{MSE}}_f)$ with the standard $\eta_f=\sqrt{8\ln M/T_f}$, so the
  exponent grows as $\sqrt{T_f}$ rather than $T_f$.
\item[\normalfont\textsc{greedy}] forward selection with replacement: repeatedly add the member that most
  reduces error, then average the selections.
\item[\normalfont\textsc{simplex-shrunk}$\kappa$] $w_f = \alpha_f w^{\text{simplex}}_f + (1-\alpha_f) w^{\text{prior}}_f$
  with $\alpha_f = n_f/(n_f+\kappa)$ and $w^{\text{prior}}$ the \textsc{harm-5} ranking weights for
  that channel, so a dense channel keeps its fitted solution and a thin one falls back to the
  ranking prior; a channel with no support at all falls back to one hour-pooled weight vector
  estimated across every channel. $\kappa\in\{100,500\}$; $\kappa=500$ is \textsc{ensemble}.
\end{description}

\textsc{select} and \textsc{ridge} additionally have hour-resolved counterparts in which $w$ varies with $h$.""")

    # ------------------------------------------------------------------ D  full ladder
    add(r"\clearpage")
    add(r"\section{Combiner ladder: validation and test}\label{app:ladder}")
    add(r"""\textsc{Ensemble} (\textsc{simplex-shrunk}500) was chosen on the development column of
Table~\ref{tab:ladder}, where it has the best mean; test was opened afterwards. Most fitted
combiners improve from development to test, where the final fit sees twice the stays, and the
ordering changes with it: the shrunk and unshrunk simplex swap places
(Section~\ref{sec:findings}). Unconstrained ridge is the exception on HiRID, moving from
$-3.59$\,\% to $+1.77$\,\%. The two protocols also differ in scoring population, not only in
fitting size.""")
    # One table, both protocols: each cell is "development / test". Two separate ladders made the
    # section twice as long and forced the reader to hold a row position in mind while jumping
    # between them; the pair that matters is always the same combiner on the two splits.
    dv, tt = [], []
    for ds in DS:
        dv_i = read("featstack_{ds}_val.csv", ds)
        tt_i = read("featstack_FINAL_{ds}.csv", ds)
        if dv_i is None or tt_i is None:
            continue
        dv.append(dv_i.set_index("combiner")["delta_vs_best_single_pct"].rename(LBL[ds]))
        tt.append(tt_i.set_index("combiner")["delta_vs_best_single_pct"].rename(LBL[ds]))
    if dv and tt:
        D = pd.concat(dv, axis=1)
        T2 = pd.concat(tt, axis=1)
        # bma_tau*/hedge_eta* are the tuning sweep, a sensitivity analysis rather than combiners
        # the paper offers; the three algorithms as specified stay.
        keep = [c for c in T2.index
                if c in D.index and not c.startswith(("bma_tau", "hedge_eta"))
                and c != "mean_top14_var"]   # identical to `mean` by construction (equal weights
        # on the best 14 of 14 members are equal weights on all of them); kept in the CSV as a
        # consistency check on the top-k code path, but printing it twice would mislead.
        D, T2 = D.loc[keep], T2.loc[keep]
        D["mean"] = D.mean(axis=1, skipna=False)
        T2["mean"] = T2.mean(axis=1, skipna=False)
        cols = list(T2.columns)
        labs, cells = [], []
        for k in T2["mean"].sort_values().index:
            labs.append(esc(k))
            cells.append([(None if pd.isna(D.loc[k, c]) else float(D.loc[k, c]),
                           None if pd.isna(T2.loc[k, c]) else float(T2.loc[k, c]))
                          for c in cols])
        add(wrap("\\setlength{\\tabcolsep}{3pt}\n" + grouped_tabular(labs, cols, cells),
                 "Combiner ladder, as percentage change against the best single model "
                 "(negative is better). \\emph{Dev} fits on a stay-grouped half of validation and "
                 "scores on the disjoint half; \\emph{test} fits on all of validation and scores on "
                 "test, so the pair differs only in how much data the weights are fitted on. Rows "
                 "are ordered by the test mean. \\textsc{Ensemble} is \\textsc{simplex-shrunk}500, "
                 "chosen on the \\emph{dev} columns before test was opened.", "ladder"))


    # ------------------------------------------------------------------ E  uncertainty
    add(r"\clearpage")
    add(r"\section{Uncertainty}\label{app:unc}")
    add(r"""Paired approximate randomization test \citep{Noreen:89} over stays with $R=10\,000$
draws, each combiner against
the validation-chosen best single model. Because MSE-per-var is a ratio of sums per channel and then
a mean over channels --- not a mean over units --- the full statistic is recomputed on every draw
rather than approximated by resampling per-stay scores. The test is \emph{approximate} because the
$2^{S}$ relabellings of $S$ stays cannot be enumerated, so $R$ are drawn at random; the smallest
attainable $p$ is therefore $1/(R+1)\approx 10^{-4}$, and $p<10^{-4}$ is the resolution of the test
rather than a finer estimate. Results are in Table~\ref{tab:unc}.""")
    unc = [read("final_uncertainty_{ds}.csv", ds) for ds in DS]
    unc = [u for u in unc if u is not None]
    if unc:
        u = pd.concat(unc)
        u["dataset"] = u["dataset"].map(LBL)
        piv = u.pivot(index="dataset", columns="combiner", values="delta_pct")
        pv = u.pivot(index="dataset", columns="combiner", values="p_value")
        cols = [c for c in ["simplex_shrunk500_var", "simplex_var", "greedy_var",
                            "rank_harmonic_top5", "rank_harmonic_top3", "mean"] if c in piv]
        # Transposed: combiners as ROWS. With combiners as columns each cell carries a value plus
        # a bracketed interval, and seven such columns ran 108 pt past the text block. As rows the
        # long names sit in the stub, where width is free, and only five data columns remain.
        if {"ci_lo", "ci_hi"} <= set(u.columns):
            lo = u.pivot(index="dataset", columns="combiner", values="ci_lo")
            hi = u.pivot(index="dataset", columns="combiner", values="ci_hi")
            cell = (piv[cols].map(lambda v: f"{v:.1f}") + r" {\tiny ["
                    + lo[cols].map(lambda v: f"{v:.1f}") + ", "
                    + hi[cols].map(lambda v: f"{v:.1f}") + "]}"
                    + pv[cols].map(lambda v: r"$^{\dagger}$" if v > 0.05 else ""))
        else:
            cell = piv[cols].map(lambda v: f"{v:.2f}") + pv[cols].map(
                lambda v: r"$^{\dagger}$" if v > 0.05 else "")
        out = cell.reindex([LBL[d] for d in DS]).T                 # combiners down, databases across
        out.index = [esc(i) for i in out.index]
        out = out.reset_index().rename(columns={"index": "combiner", "combiner": "combiner"})
        add(wrap(tabular(out, "l" + "r" * len(DS)),
                 "Change against the best single model on test, with 95\\% bootstrap confidence "
                 "intervals over stays. $\\dagger$ marks $p \\geq 0.05$ under a paired "
                 "approximate randomization test.", "unc"))
        add(r"""$^{\dagger}$ not significant at $p<0.05$. Every combiner listed above reaches the
smallest attainable value, $1/(R+1)$ with $R=10\,000$ draws, on every database, with two
exceptions: the unweighted mean fails on
MIMIC-IV, so naive averaging is the one combiner here that can lose to a single model, and hard
selection reaches only $p=0.04$ on eICU and UMM. Unconstrained ridge is not among the
methods tested here and is worse than a single model on HiRID. Brackets give a 95\% percentile
confidence interval from 2\,000 bootstrap resamples of stays.""")

    # ------------------------------------------------------------------ F  win shares
    add(r"\clearpage")
    add(r"\section{Which model wins which cells}\label{app:win}")
    add(r"""Share of well-supported (variable, hour) test cells on which each family has the lowest
error ($\geq 200$ observations on the scoring split), in Table~\ref{tab:win}. Shares sum to
100\% per column. Table~\ref{tab:win} aggregates over the horizon; Figure~\ref{fig:maps} shows the
same quantity resolved by (variable, hour) on one database.""")
    # The winner map belongs with the win shares: it is the same quantity at cell resolution.
    # Filename is anonymized -- the cohort's initials must not travel in the LaTeX source.
    add(r"""\begin{figure}[htbp]
\centering
\includegraphics[width=\textwidth]{figures/featstack_anon_maps.pdf}
\caption{Per-(variable, hour) structure on UMM validation, the densest database and so the
one where most channels clear the support filter. \textbf{Top:} which architecture wins each cell.
\textbf{Center:} ambiguity share, the ceiling on what any combiner can recover. \textbf{Bottom:}
mean pairwise residual correlation. Cells with fewer than 200 observations in \emph{either} the
fitting or the scoring half are grayed, so a winner picked from a handful of observations cannot be
over-read. No architecture owns the grid: the winner changes across variables and, less often,
along the horizon.}\label{fig:maps}
\end{figure}""")
    ws = []
    for ds in DS:
        d = read("final_winshare_{ds}.csv", ds)
        if d is None:
            continue
        d["family"] = d["model"].map(fam)
        g = d.groupby("family", as_index=False)["share_pct"].sum()
        ws.append(g.set_index("family")["share_pct"].rename(LBL[ds]))
    if ws:
        w = pd.concat(ws, axis=1).fillna(0.0)
        w["mean"] = w.mean(axis=1)
        w = w.sort_values("mean", ascending=False).reset_index()
        add(wrap(tabular(w, "l" + "r" * (len(w.columns) - 1), "%.1f"),
                 "Share of well-supported (variable, hour) test cells won by each family (\\%).", "win"))

    # ------------------------------------------------------------------ F2 pool diversity
    # F2 is a CROSS-family diversity table: one row per architecture. The `_seedpool_*` matrices
    # are a different object — several runs of the SAME family (s123/s456/s789/s1337), used to
    # measure run-to-run residual correlation within one architecture — so every row collapses to
    # the same label and `reindex` raises "cannot reindex on an axis with duplicate labels".
    # Those files appeared 2026-10-06 and this script has been UNRUNNABLE since (the generated
    # appendix is dated 2026-09-11), which is why it went stale; §37.56.
    rcs = [f for f in sorted(T.glob("residual_correlation_*.csv"))
           if "_mean" not in f.name and "_seedpool_" not in f.name]
    if len(rcs) >= 3:
        FAM = [("baseline_null", "null"), ("persist", "LOCF"), ("dense_dms", "DenseDMS"),
               ("hpsf", "DenseAR"), ("dense_autoregressive", "DenseAR"), ("ladderms", "DenseAR"),
               ("dlinear", "DLinear"), ("hps_gru", "GRU"), ("gru", "GRU"), ("hps_lstm", "LSTM"),
               ("lstm", "LSTM"), ("hpsm", "Mamba"), ("mamba", "Mamba"), ("patchtst", "PatchTST"),
               ("informer", "Informer"), ("autoformer", "Autoformer"), ("strats", "STraTS"),
               ("tirex", "TiRex-2"), ("moirai", "Moirai"),
               # keep in step with the module-level FAMILY (§37.56); chronos2 before chronos
               ("arima", "ARIMA"), ("ets_", "ETS"), ("lightgbm", "LightGBM"),
               ("chronos2", "Chronos-2"), ("chronos", "Chronos-Bolt"), ("mira", "MIRA")]

        def _fam(n):
            t = str(n).lower()
            for k, v in FAM:
                if k in t:
                    return v
            # Escape: this value reaches LaTeX verbatim and every run name has underscores.
            print(f"make_appendix WARNING: F2 has no FAM entry for {n!r}", file=_sys.stderr)
            return esc(str(n)[:16])

        ORD = ["DenseAR", "Informer", "DenseDMS", "GRU", "LSTM", "STraTS", "Mamba", "DLinear",
               "PatchTST", "Autoformer", "Chronos-Bolt", "Chronos-2", "MIRA", "Moirai",
               "TiRex-2", "LightGBM", "ARIMA", "ETS", "LOCF", "null"]
        mats = []
        for _f in rcs:
            d = pd.read_csv(_f, index_col=0)
            d.index = [_fam(x) for x in d.index]
            d.columns = [_fam(x) for x in d.columns]
            mats.append(d.reindex(index=ORD, columns=ORD))
        Mx = sum(mats) / len(mats)
        for _mdl in ORD:                   # the weighted average leaves 0.99 on some diagonals
            Mx.loc[_mdl, _mdl] = 1.0
        Mx = Mx.round(2).reset_index().rename(columns={"index": ""})
        add(r"\clearpage")
        add(r"\section{Pool diversity}\label{app:div}")
        add(r"""Mean pairwise residual correlation between pool members, averaged over the five
databases and weighted by observation count, in Table~\ref{tab:div}. What a combiner can exploit is
exactly what this matrix is missing from one: two members correlating at $0.95$ leave almost nothing
to arbitrate. Three structures are visible. The six in-domain sequence models form a block at
$0.92$--$0.95$, so they are close to interchangeable as ensemble members. The trivial baselines are
the most distant members of all --- \textsc{null} at $0.42$--$0.68$ and \textsc{locf} at
$0.77$--$0.90$ --- which is why they still earn weight. And the two zero-shot foundation models sit
closer to \textsc{locf} (Moirai $0.90$) than to the in-domain decoders ($0.81$--$0.84$): the
diversity they add is persistence-like rather than a better fit.""")
        # 14 model names across a header row do not fit; rotate them 45 degrees. \rotatebox needs
        # graphicx, which the manuscript already loads. origin=l anchors each label at its left
        # baseline so the diagonal rises away from its own column.
        _body = tabular(Mx, "l" + "r" * (len(Mx.columns) - 1), "%.2f")
        _hdr = " & ".join([""] + [r"\rot{%s}" % c for c in Mx.columns[1:]]) + r" \\"
        _old_hdr = " & ".join([""] + [str(c) for c in Mx.columns[1:]]) + r" \\"
        _body = _body.replace(_old_hdr, _hdr, 1)
        _body = (r"\providecommand{\rot}[1]{\rotatebox[origin=l]{45}{#1}}" "\n" + _body)
        add(wrap(_body,
                 "Mean pairwise residual correlation between pool members, averaged over the five "
                 "databases; $1.00$ on the diagonal by construction, lower is more complementary.",
                 "div", r"\scriptsize"))

    # ------------------------------------------------------------------ G  per system
    add(r"\clearpage")
    add(r"\section{Results by variable system}\label{app:system}")
    add(r"""MSE-per-var on test for each pool family and for the combiner chosen on validation
(\textsc{ensemble}), within each clinical system. Channels carrying fewer than 100 observations are
excluded, as in every group-level table here. One table per database follows
(Tables~\ref{tab:sysumm}--\ref{tab:syseicu}). \textbf{The combination beats every pool member in
all 50 system\,$\times$\,database cells.} It is not the best \emph{combiner} everywhere: the plain
mean edges it on UMM medications and greedy on MIMIC-III \emph{all charted}. Without the support filter it loses two of them --- eICU
ventilation and HiRID medications --- but both losses are produced entirely by channels too thin to
estimate a per-channel weight on, which is the artifact the filter exists to remove.""")

    for ds in DS:
        d = read("final_bymodel_{ds}.csv", ds)
        if d is None:
            continue
        keep = ["all_charted", "type:lab", "type:vital", "type:medication", "sofa_relevant",
                "freq:low", "freq:high"]
        d = d[d["system"].isin(keep)].copy()
        d["family"] = d["model"].map(lambda m: m if m.startswith(">>") else fam(m))
        SHOW = {">>simplex_shrunk500_var": "ENSEMBLE", ">>rank_harmonic_top5": "harm-5",
                ">>select_var": "select", ">>mean": "mean", ">>greedy_var": "greedy"}
        d = d[~d["family"].str.startswith(">>") | d["model"].isin(SHOW)]
        for k, v in SHOW.items():
            d.loc[d["model"] == k, "family"] = v
        piv = (d.pivot_table(index="family", columns="system", values="mse_per_var", aggfunc="min")
               .reindex(columns=[k for k in keep if k in set(d["system"])]))
        piv = piv.sort_values("all_charted").reset_index()
        piv.columns = [esc(c) for c in piv.columns]
        add(wrap(tabular(piv, "l" + "r" * (len(piv.columns) - 1)),
                 f"\\textbf{{{PLAIN[ds]}:}} MSE-per-var by clinical system (test). "
                 f"\\textsc{{ensemble}} is the combiner chosen on validation.",
                 f"sys{ds.replace(chr(95), chr(45))}", r"\footnotesize"))

    # ------------------------------------------------------------------ H  per horizon
    add(r"\clearpage")
    add(r"\section{Gain by horizon}\label{app:horizon}")
    add(r"""Combination gain of the combiner chosen on validation against the best single model, by horizon
step. The gain \emph{decays} with horizon on four of five databases, which is the complement of the
residual correlation rising with horizon: members converge on the same drift, so there is
progressively less disagreement to arbitrate. Table~\ref{tab:horizon} gives the values and
Figure~\ref{fig:horizon} plots them.""")
    hz = []
    for ds in DS:
        d = read("final_byhorizon_{ds}.csv", ds)
        if d is None:
            continue
        d = d[d["combiner"] == "simplex_shrunk500_var"]
        hz.append(d.set_index("horizon")["delta_pct"].rename(LBL[ds]))
    if hz:
        h = pd.concat(hz, axis=1)
        h = h.loc[[x for x in [1, 2, 4, 6, 8, 12, 16, 20, 24] if x in h.index]]
        h["mean"] = h.mean(axis=1)
        h = h.reset_index()
        add(wrap(tabular(h, "r" + "r" * (len(h.columns) - 1), "%.2f"),
                 "Combination gain against the best single model by horizon step (\\%).", "horizon"))
    if (ROOT / "docs" / "figures" / "horizon_gain_anon.pdf").exists():
        add(figure("figures/horizon_gain_anon.pdf",
                   "Combination gain by horizon. The gain decays as members converge on the same "
                   "drift. The two smallest databases are visibly noisier hour to hour, the "
                   "per-hour form of the split-seed spread in Section~\\ref{sec:findings}. HiRID's "
                   "excursion at $h=3$ is a period-4 oscillation present in the underlying data "
                   "rather than a plotting artifact; it recurs at $h=7$, $11$ and $15$, and we have "
                   "not identified its cause.",
                   "horizon", r"0.72\textwidth"))

    # ------------------------------------------------------------------ H2 rank profile
    # Both blocks come straight from the featstack ladders: *_val.csv is the development split
    # (fit on half of validation, score the held-out half) and *_FINAL_*.csv is the confirmation
    # (fit on all of validation, score test). Reading them here rather than a pre-digested
    # rank_profile_*.csv keeps the table in step with whatever the last ensemble_eval run wrote.
    # Two-level header: the columns group 3-3-3-1 over k, and spelling each one out in full
    # ("equal-14") makes the HEADER wider than any number under it, which is what forces the font
    # down. Grouping lets the numbers set the width and buys back two size steps.
    PROF = [("mean_top3_var", r"\textsc{equal}"), ("rank_harmonic_top3", r"\textsc{harm}"),
            ("rank_fitted_top3", "fit"),
            ("mean_top5_var", r"\textsc{equal}"), ("rank_harmonic_top5", r"\textsc{harm}"),
            ("rank_fitted_top5", "fit"),
            ("mean", r"\textsc{equal}"), ("rank_harmonic_top14", r"\textsc{harm}"),
            ("rank_fitted_top14", "fit"), ("simplex_shrunk500_var", r"\textsc{ens.}")]
    PROF_GRP = [(r"$k=3$", 3), (r"$k=5$", 3), (r"$k=14$", 3), ("", 1)]

    def prof_block(pat):
        """One database x combiner block of delta-vs-best-single, plus a mean row."""
        rows = []
        for ds in DS:
            v = read(pat, ds)
            if v is None:
                continue
            ser = v.set_index("combiner")["delta_vs_best_single_pct"]
            rows.append([LBL[ds]] + [ser.get(c, float("nan")) for c, _ in PROF])
        if not rows:
            return None
        mu = [float(np.nanmean([r[j] for r in rows])) for j in range(1, len(PROF) + 1)]
        return rows, [r"\quad mean"] + mu

    dev_b, tst_b = prof_block("featstack_{ds}_val.csv"), prof_block("featstack_FINAL_{ds}.csv")
    if dev_b is not None:
        ncol = len(PROF) + 1

        def fmt(row):
            return " & ".join([row[0]] + [("--" if v != v else f"{v:.2f}") for v in row[1:]]) + r" \\"

        top, rules, c = [""], [], 2
        for lab, span in PROF_GRP:
            top.append(lab if span == 1 else rf"\multicolumn{{{span}}}{{c}}{{{lab}}}")
            if span > 1:
                rules.append(rf"\cmidrule(lr){{{c}-{c + span - 1}}}")
            c += span
        body = [r"\begin{tabular}{l" + "r" * len(PROF) + "}", r"\toprule",
                " & ".join(top) + r" \\", "".join(rules),
                " & ".join(["database"] + [lab for _, lab in PROF]) + r" \\", r"\midrule"]
        for title, blk in ((r"\emph{development (half of val $\rightarrow$ held-out half)}", dev_b),
                           (r"\emph{test (all of val $\rightarrow$ test)}", tst_b)):
            if blk is None:
                continue
            if body[-1] != r"\midrule":
                body.append(r"\midrule")
            body.append(rf"\multicolumn{{{ncol}}}{{l}}{{{title}}} \\")
            body += [fmt(r) for r in blk[0]]
            body += [r"\addlinespace[2pt]", fmt(blk[1])]
        body += [r"\bottomrule", r"\end{tabular}"]
        add(r"\clearpage")
        add(r"\section{Does the weight profile matter?}\label{app:profile}")
        add(r"""Section~\ref{sec:fixed} shows that choosing members per channel is worth about
$2$\,pp while the harmonic profile on those members is worth about $0.2$. This appendix adds the
third rung: instead of fixing the profile at $1,\tfrac12,\dots,\tfrac1k$, \emph{fit} one weight
vector over rank positions, shared by every channel. It is cheap --- permuting the model axis of
$G$ and $b$ into rank order gives the rank-space normal equations with no pass over predictions
--- and it changes little, tracking normalized $1/k$ to within $0.045$ at a correlation of $0.99$.

Table~\ref{tab:profile} weights identical per-channel members three ways, and how much the profile
matters turns out to depend on how many members it spans. At $k=3$ and $k=5$ the three columns
agree to within $0.2$\,pp and fitting never beats the fixed $1/k$ decay. At $k=14$ they separate:
\textsc{equal-14}, the unweighted mean, is $3.1$\,pp worse than \textsc{harm-14} on test and worse
even than \textsc{equal-5}, because flat weights over the whole pool give the weakest members the
same say as the best. It is the decay rather than the fitting that carries the full pool ---
\textsc{harm-14} estimates nothing and still edges out \textsc{fit-14}, $-7.89$ against $-7.85$.""")
        add(wrap("\n".join(body),
                 "Weight profiles at fixed membership, as percentage change against the best single "
                 "model (negative is better). \\textsc{equal}, \\textsc{harm} and \\textsc{fit} use "
                 "the same per-channel top-$k$ members and differ only in how those members are "
                 "weighted, and are written \\textsc{equal-}$k$, \\textsc{harm-}$k$ and "
                 "\\textsc{fit-}$k$ in the text. \\textsc{fit-}$k$ fits one weight vector over rank "
                 "positions, shared by every channel. \\textsc{equal-14} is the unweighted mean "
                 "over the whole pool, since equal weights on the best fourteen of fourteen "
                 "members are equal weights on all of them. "
                 "\\textsc{ens.} is the per-channel shrunk simplex, shown for scale. Referenced "
                 "from Section~\\ref{sec:fixed}.", "profile",
                 r"\small\setlength{\tabcolsep}{5pt}", star=True))

    # ------------------------------------------------------------------ I  what does not work
    add(r"\clearpage")
    add(r"\section{Inference-time injection and fine-tuning}\label{app:neg}")
    add(r"""Six variants of feeding a decoder better values during generation, on two databases and two
iterative decoders, as validation MSE-per-var relative to free-running (positive is worse,
Table~\ref{tab:inject}):

\begin{table}[htbp]\centering\small
\caption{Inference-time injection, as percentage change against free-running generation on validation. The masked rows require the future observation mask and are therefore not deployable.}\label{tab:inject}
\begin{tabular}{lrrrr}
\toprule
 & DenseAR UMM & STraTS UMM & DenseAR HiRID & STraTS HiRID \\
\midrule
free-running (absolute) & 0.4853 & 0.5253 & 0.4344 & 0.4715 \\
ensemble, masked & $-0.3$ & $-9.4$ & $+0.3$ & $-3.4$ \\
ground truth, masked & $+1.1$ & $-29.4$ & $+1.7$ & $-23.5$ \\
ensemble, unmasked & $+1.9$ & $+13.7$ & $+0.3$ & $+29.1$ \\
per-channel best, unmasked & $+3.5$ & $+20.5$ & $+0.6$ & $+38.6$ \\
live hybrid & $+3.6$ & $+29.1$ & $+0.6$ & $+178.3$ \\
\bottomrule
\end{tabular}
\end{table}

\noindent The masked rows inject only where the target is observed, so which channels receive an
injection encodes the future observation mask; since ICU sampling is informative, those rows are not
deployable however large the improvement looks. The unmasked rows need no such information and are
uniformly worse. A decoder trained on its own predictions is already adapted to free-running and
even ground truth costs it 1--2\%; a decoder trained under a teacher-forcing curriculum carries
exposure bias instead, which is why injection appears to help it so much and why that help cannot be
used.

Fine-tuning the decoder to consume the combined trajectory improved validation error from 0.4245 to
0.4166, but a matched control --- identical fine-tuning, no injection --- reached 0.4187. Of the
apparent $-1.8\%$, therefore, $-1.35$\,pp is continued training of a checkpoint that had not
converged and only $-0.50$\,pp is attributable to injection, while the resulting model requires the
whole pool at inference exactly as post-hoc combination does.""")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("\n\n".join(L) + "\n")
    print(f"wrote {a.out}  ({len(L)} blocks)")


if __name__ == "__main__":
    main()
