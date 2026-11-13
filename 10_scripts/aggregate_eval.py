"""
10_scripts/aggregate_eval.py
Aggregate per-run downstream clinical evaluation JSONs (docs/tables/eval_*.json,
written by harmonize.eval_forecast) into comparison tables across
model × dataset × score.

Highlights the ΔSOFA≥2 organ-dysfunction detection sensitivity — the key
"can the forecast detect deterioration?" metric — alongside forecast error,
SOFA MAE, SAPS-II MAE, and (where computed) Sepsis-3 detection.

Usage (from project root):
  python 10_scripts/aggregate_eval.py [--glob 'docs/tables/eval_*.json']
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _flatten(d: dict) -> dict:
    fc, sofa, saps, sep = (d.get("forecast", {}), d.get("sofa", {}),
                           d.get("sapsii", {}), d.get("sepsis3", {}))
    aki = d.get("aki", {})
    run = d.get("run_name", "")
    nested = {"forecast", "sofa", "sapsii", "sepsis3", "aki"}
    # d_model: prefer the checkpoint's config.json (reliable for the neural models;
    # the eval JSON itself carries no d_model), fall back to parsing the run name.
    dmodel = None
    cfg = ROOT / "checkpoints" / run / "config.json"
    if cfg.exists():
        try:
            dmodel = json.loads(cfg.read_text()).get("d_model")
        except Exception:
            dmodel = None
    if dmodel is None:
        m = re.search(r"d(\d+)", run)
        dmodel = int(m.group(1)) if m else None
    return {
        "run_name":   run,
        "dataset":    d.get("dataset", ""),
        "model":      d.get("model", ""),
        "d_model":    dmodel,
        "n_windows":  d.get("n_windows"),
        # forecast error (test)
        "fc_mse_per_var": fc.get("val_mse_per_var"),
        "fc_mae_per_var": fc.get("val_mae_per_var"),
        # SOFA
        "sofa_total_mae": sofa.get("sofa_max_mae"),
        "sofa_resp_mae":  sofa.get("sofa_resp_mae"),
        "sofa_renal_mae": sofa.get("sofa_renal_mae"),
        "dSOFA2_auroc":   sofa.get("dSOFA2_auroc"),
        "dSOFA2_sens":    sofa.get("dSOFA2_sensitivity"),
        "dSOFA2_spec":    sofa.get("dSOFA2_specificity"),
        "dSOFA2_prev":    sofa.get("dSOFA2_gt_prevalence"),
        # Daily ΔSOFA: daily_sofa(target) − daily_sofa(input), per-organ WORST on BOTH sides,
        # i.e. the same quantity as the stored dsofa2 head label. The "wb" in the key is a
        # historical misnomer for "worst−best" — a min/"best" baseline is NOT what is computed
        # and would give ~52 % positives instead of ~12 % (§37.52 D31). The key name is kept
        # because it ships in all 526 eval tables; only the description is corrected.
        "dSOFA2wb_auroc": sofa.get("dSOFA2wb_auroc"),
        "dSOFA2wb_sens":  sofa.get("dSOFA2wb_sensitivity"),
        "dSOFA2wb_prev":  sofa.get("dSOFA2wb_gt_prevalence"),
        # AKI (KDIGO) — creatinine-only + urine-gated, forecast detection
        "aki_auroc":      aki.get("aki_auroc"),
        "aki_sens":       aki.get("aki_sensitivity"),
        "aki_prev":       aki.get("aki_gt_prevalence"),
        "aki_uo_auroc":   aki.get("aki_uo_auroc"),
        "aki_uo_sens":    aki.get("aki_uo_sensitivity"),
        # SAPS-II
        # GCS-free variants + SAPS-II coverage (§37.39/§37.41): the GCS-bearing tables cover the
        # four databases with usable GCS, these cover all five. `*_components_scored_mean` is why
        # a SAPS-II number must never be read alone — a complete SAPS-II exists in only 3.6-20.3 %
        # of windows.
        "sofa_noCNS_total_mae":  sofa.get("noCNS_sofa_max_mae"),
        "sofa_noCNS_dsofa_mae":  sofa.get("noCNS_delta_sofa_mae"),
        "sapsii_noGCS_mae":      saps.get("sapsii_noGCS_mae"),
        "sapsii_noGCS_gt_mean":  saps.get("sapsii_noGCS_gt_mean"),
        "sapsii_comp_scored":    saps.get("sapsii_components_scored_mean"),
        "sapsii_all_comp_frac":  saps.get("sapsii_all_components_frac"),
        "sapsii_gcs_obs_frac":   saps.get("sapsii_gcs_obs_frac"),
        "sapsii_mae":     saps.get("sapsii_mae"),
        # Sepsis-3 (optional)
        "sepsis3_sens":   sep.get("sensitivity"),
        "sepsis3_spec":   sep.get("specificity"),
        "sepsis3_prev":   sep.get("gt_prevalence"),
    } | {
        # Pass through any further top-level scalar. Without this the fixed eval_* schema
        # above silently drops every key it does not name, so a different JSON family
        # (lgbmhead's val/test AUROC, head, seed, ...) aggregates to an EMPTY table --
        # which is worse than an error, because a useless CSV still gets written.
        k: v for k, v in d.items()
        if k not in nested and not isinstance(v, (dict, list))
        and k not in {"run_name", "dataset", "model", "n_windows"}
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="docs/tables/eval_*.json")
    ap.add_argument("--out", default="docs/tables/eval_comparison.csv")
    args = ap.parse_args()

    files = sorted(ROOT.glob(args.glob.replace("docs/", "docs/")))
    if not files:
        # allow absolute or relative glob
        files = sorted(Path(".").glob(args.glob))
    if not files:
        print(f"No eval JSONs found matching {args.glob}")
        return

    rows = []
    for f in files:
        try:
            r = _flatten(json.loads(f.read_text()))
            # the stem carries what the JSON body does not (e.g. the lgbmhead arm
            # window/traj/trajpm), so keep it: every result CSV stays self-describing.
            r.setdefault("source", f.stem)
            rows.append(r)
        except Exception as e:
            print(f"  skip {f.name}: {e}")
    df = pd.DataFrame(rows)
    # sort on whichever of the usual keys this JSON family actually has, so the script
    # works for any docs/tables/*.json family, not just eval_* (repository policy:
    # results are published as CSV, so every family needs a path through here).
    by = [c for c in ("dataset", "model", "d_model", "head", "source") if c in df.columns]
    if by:
        df = df.sort_values(by)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Loaded {len(df)} eval runs → {out}\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", "{:.3f}".format)

    print("=" * 70)
    print("  FORECAST ERROR + SOFA (per run)")
    print("=" * 70)
    cols1 = ["dataset", "model", "d_model", "fc_mse_per_var",
             "sofa_total_mae", "dSOFA2_sens", "dSOFA2_spec", "dSOFA2_prev"]
    print(df[[c for c in cols1 if c in df]].to_string(index=False))

    if df["sepsis3_sens"].notna().any():
        print("\n" + "=" * 70)
        print("  SEPSIS-3 DETECTION (where computed)")
        print("=" * 70)
        s = df[df["sepsis3_sens"].notna()]
        print(s[["dataset", "model", "sepsis3_sens", "sepsis3_spec",
                 "sepsis3_prev"]].to_string(index=False))

    # Key finding table: ΔSOFA sensitivity by dataset × model
    print("\n" + "=" * 70)
    print("  KEY: ΔSOFA≥2 SENSITIVITY (deterioration detection) — model × dataset")
    print("=" * 70)
    try:
        piv = df.pivot_table(index="model", columns="dataset",
                             values="dSOFA2_sens", aggfunc="mean")
        print(piv.to_string())
    except Exception as e:
        print(f"  (pivot failed: {e})")

    print(f"\nFull table: {out}")


if __name__ == "__main__":
    main()
