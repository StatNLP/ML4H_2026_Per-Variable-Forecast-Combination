#!/bin/bash
# jobs/run_final_test.sh — THE FINAL CONFIRMATION. Fit on ALL of val, score on ALL of test.
#
# This spends the held-out set. Run it ONCE, after the combiner is chosen; anything tuned after
# seeing these numbers is no longer held out.
#
# Fitting on the whole val split (rather than a stay-grouped half) also removes the fragility
# §27.5 documents: UMM's gain ranged -11.15%..-1.17% across half-splits because ~355 stays cannot
# pin down 94 variables x 14 models. Here UMM fits on all 710 val stays and is scored on test's
# 891 — no arbitrary partition, so no partition seed to be sensitive to.
#
#   sbatch jobs/run_final_test.sh                       # val -> test, all five
#   MODE=dev SWEEP=1 sbatch jobs/run_final_test.sh      # val-half -> val-half, + knob sweep
#
# MODE=dev keeps the development protocol (stay-grouped half of val, scored on the disjoint half)
# so anything tuned there never sees test. SWEEP=1 adds the BMA effective-sample-size discounts and
# the Hedge learning rates: run the sweep in dev, decide there, then spend test once.
#SBATCH --job-name=final_test
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/final_test_%j.out
#SBATCH --error=logs/final_test_%j.err
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"; cd "$ROOT"; export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PY="${PYTHON:-python}"
# =====================================================================================
# REBUTTAL ADD-ONS (ML4H 2026 reviews). Placed BEFORE the main loop and ending in exit, so they can
# never fall through into a final-test rerun -- the first version sat after the loop and re-ran the
# published evaluation before reaching its own block. DEV protocol only: fit on a stay-grouped half
# of validation, score on the disjoint half; test was spent once and these are NEW estimators.
#
# NOFM=1        the manifest pool minus Moirai and TiRex-2 (pmk4, 6g9V: pretraining exposure).
# SEEDPOOL=strats | dense_autoregressive
#               SbLR's variance control: N seeds of ONE architecture as the whole pool. One run per
#               architecture -- mixing two families' seeds would not be the control he asked for.
# WIDE=1        the full 14-model pool PLUS extra members, ADD="arima ets lightgbm" by default
#               (JFiH: pool homogeneity). ADD="arima ets" = without LightGBM; "lightgbmtr" = the
#               LightGBM variant early-stopped on held-out TRAIN stays (leakage control).
# POOL14=1      the published 14-model pool, re-run only for the extra tables / DEVARGS sweeps.
# KABL="2 4 ..." model-count ablation (BenJ): DRAWS random k-subsets of the 14-model pool per k.
# DEVARGS       passed through to ensemble_eval (e.g. "--ridge-lambda 10"); RBTAG overrides the tag.
if [[ "${NOFM:-0}" == "1" || -n "${SEEDPOOL:-}" || "${WIDE:-0}" == "1" || "${POOL14:-0}" == "1" || -n "${KABL:-}" ]]; then
  if   [[ "${NOFM:-0}" == "1" ]]; then TAG=nofm
  elif [[ "${WIDE:-0}" == "1" ]]; then TAG=wide
  elif [[ "${POOL14:-0}" == "1" ]]; then TAG=p14
  elif [[ -n "${KABL:-}" ]]; then TAG=kabl
  else TAG="seedpool_${SEEDPOOL}"; fi
  TAG=${RBTAG:-$TAG}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name="rb_${TAG}" 2>/dev/null || true
  # The seed dump names its output by checkpoint, and one published pool member IS a seed
  # checkpoint (strats_mimic_iii_ms_s789), so the dump rewrites it. The originals were copied to
  # data/base_preds_protected/ with checksums before the dump reached them. Restore on mismatch so
  # the published pool's predictions are bit-identical to the ones the paper was scored on.
  if [[ -f data/base_preds_protected/SHA256SUMS ]]; then
    if sha256sum --quiet -c data/base_preds_protected/SHA256SUMS; then
      echo "protected pool members unchanged"
    else
      echo "!! protected pool members changed by the seed dump -- restoring originals"
      cp -p data/base_preds_protected/*.npy data/base_preds/
      sha256sum -c data/base_preds_protected/SHA256SUMS || { echo "RESTORE FAILED"; exit 1; }
    fi
  fi
  val_path() {  # $1 member, $2 dataset -> val prediction path (protected copy preferred)
    local n=$1 d=$2
    case $n in tirex2|moirai_small) echo "data/fm_preds/${n}_${d}_val.npy"; return;; esac
    [[ -f data/base_preds_protected/${n}__${d}_val.npy ]] && { echo data/base_preds_protected/${n}__${d}_val.npy; return; }
    echo data/base_preds/${n}__${d}_val.npy
  }
  run_dev() {  # $1 dataset, $2 tag, rest = member names
    local d=$1 t=$2; shift 2
    local VP="" n pv
    for n in "$@"; do pv=$(val_path "$n" "$d")
      [[ -f "$pv" ]] || { echo "  !! $n missing on val -- left out"; continue; }
      VP="$VP ${n}=$pv"; done
    local k=$(echo $VP | wc -w)
    [[ $k -lt 2 ]] && { echo "skip $d/$t: $k members"; return; }
    echo "=== ${t} $d ($k members) $(date) ==="
    $PY -m harmonize.ensemble_eval --mode featstack --test-h5 "data/dataset/${d}_seed42_val__all_4h_fwd.h5" \
        --pred-files $VP --n-clusters 1 --table-tag "_rb_${t}" ${RBX:+--rebuttal-extras} ${DEVARGS:-} \
        || echo "  ($d/$t FAILED)"
  }
  for ds in ${DSETS:-umm hirid mimic_iii mimic_iv eicu}; do
    [[ -f "data/dataset/${ds}_seed42_val__all_4h_fwd.h5" ]] || { echo "skip $ds (no val h5)"; continue; }
    BASE=$(awk -F'\t' -v d="$ds" '$1==d{print $2}' configs/pool_manifest.tsv | sort)
    X=""; for a in ${ADD:-}; do   # extra members; NOFM uses them only when ADD is given
      case $a in arima|ets) X="$X ${a}_${ds}_4h";; *) X="$X ${a}_${ds}";; esac; done
    if [[ "${NOFM:-0}" == "1" ]]; then
      run_dev $ds $TAG $BASE $X
    elif [[ "${WIDE:-0}" == "1" || "${POOL14:-0}" == "1" ]]; then
      X=""; if [[ "${WIDE:-0}" == "1" ]]; then for a in ${ADD:-arima ets lightgbm}; do
        case $a in arima|ets) X="$X ${a}_${ds}_4h";; *) X="$X ${a}_${ds}";; esac; done; fi
      run_dev $ds $TAG $BASE $X tirex2 moirai_small
    elif [[ -n "${KABL:-}" ]]; then
      ALL14="$BASE tirex2 moirai_small"
      for k in $KABL; do for d in $(seq 1 ${DRAWS:-5}); do
        # seeded per (dataset, k, draw); `shuf --random-source=<(yes ...)` gave the SAME subset for
        # every draw, because the seed strings share a prefix and shuf reads only the first bytes
        SUB=$($PY -c "import random,sys; print(' '.join(random.Random(sys.argv[1]).sample(sys.argv[3:], int(sys.argv[2]))))" "${ds}-${k}-${d}" $k $ALL14)
        run_dev $ds "${TAG}_k${k}_d${d}" $SUB
      done; done
    else
      run_dev $ds $TAG $(ls data/base_preds/${SEEDPOOL}_${ds}_ms_s*__${ds}_val.npy 2>/dev/null \
              | sed "s|.*/||;s|__${ds}_val.npy||" | sort)
    fi
  done
  exit 0
fi

MODE=${MODE:-final}
EXTRA=""; [[ "${SWEEP:-0}" == "1" ]] && EXTRA="--combiner-sweep"
# Any run with ADD or DROPFM is NOT the published pool, so it must not write the canonical
# featstack_<ds>_val_*.csv. FTAG is mandatory in that case and tags the output instead.
TT=""; [[ -n "${FTAG:-}" ]] && TT="--table-tag _${FTAG}"
if [[ -z "${FTAG:-}" && ( -n "${ADD:-}" || "${DROPFM:-0}" == "1" ) ]]; then
  echo "refusing to run a modified pool without FTAG -- it would overwrite the published tables"; exit 1
fi
for ds in ${*:-umm hirid mimic_iii mimic_iv eicu}; do
  vh="data/dataset/${ds}_seed42_val__all_4h_fwd.h5"
  th="data/dataset/${ds}_seed42_test__all_4h_fwd.h5"
  [[ -f "$vh" && -f "$th" ]] || { echo "skip $ds"; continue; }
  # The two splits must list the SAME models in the SAME order — weights are positional, and a
  # mismatched order would silently apply every weight to the wrong model. Build both from one
  # sorted name list and let ensemble_eval re-check.
  # The pool is the MANIFEST, not "whatever is in data/base_preds". The directory also receives
  # seed dumps and other experiments' predictions; globbing it once silently added members to the
  # published 14-model pool. configs/pool_manifest.tsv pins the 12 base members per database.
  NAMES=$(awk -F'\t' -v d="$ds" '$1==d{print $2}' configs/pool_manifest.tsv | sort)
  [[ -n "$NAMES" ]] || { echo "  !! $ds has no rows in configs/pool_manifest.tsv"; continue; }
  VP=""; TP=""
  for n in $NAMES; do
    [[ -f "data/base_preds/${n}__${ds}_test.npy" ]] || { echo "  !! $n missing on test — skipping $ds"; VP=""; break; }
    pv=data/base_preds/${n}__${ds}_val.npy; pt=data/base_preds/${n}__${ds}_test.npy
    [[ -f data/base_preds_protected/${n}__${ds}_val.npy ]]  && pv=data/base_preds_protected/${n}__${ds}_val.npy
    [[ -f data/base_preds_protected/${n}__${ds}_test.npy ]] && pt=data/base_preds_protected/${n}__${ds}_test.npy
    VP="$VP ${n}=$pv"; TP="$TP ${n}=$pt"
  done
  [[ -z "$VP" ]] && continue
  # ADD extends the published pool on BOTH splits. val and test must stay in the same
  # positional order -- weights are positional -- so each member is appended to VP and TP
  # together or to neither. DROPFM=1 then leaves the two zero-shot foundation models out,
  # which is the 15-member extended pool (12 manifest + arima + ets + lightgbmfix).
  for a in ${ADD:-}; do
    case $a in arima|ets) nm="${a}_${ds}_4h";; *) nm="${a}_${ds}";; esac
    av="data/base_preds/${nm}__${ds}_val.npy"; at="data/base_preds/${nm}__${ds}_test.npy"
    [[ -f "$av" && -f "$at" ]] || { echo "  !! $nm missing on val or test -- left out"; continue; }
    VP="$VP ${nm}=$av"; TP="$TP ${nm}=$at"
  done
  if [[ "${DROPFM:-0}" != "1" ]]; then   # NOFM is taken by the dev branch above
  for fm in tirex2 moirai_small; do
    v="data/fm_preds/${fm}_${ds}_val.npy"; t="data/fm_preds/${fm}_${ds}.npy"
    [[ -f "$v" && -f "$t" ]] && { VP="$VP ${fm}=$v"; TP="$TP ${fm}=$t"; }
  done
  fi
  if [[ "$MODE" == "dev" ]]; then
    echo "=== DEV $ds  ($(echo $VP|wc -w) models)  $(date) ==="
    $PY -m harmonize.ensemble_eval --mode featstack \
        --test-h5 "$vh" --pred-files $VP \
        --n-clusters 1 $EXTRA $TT || echo "  ($ds FAILED)"
  else
    echo "=== FINAL $ds  ($(echo $VP|wc -w) models)  $(date) ==="
    $PY -m harmonize.ensemble_eval --mode featstack \
        --test-h5 "$vh" --pred-files $VP \
        --eval-h5 "$th" --eval-pred-files $TP \
        --n-clusters 1 $EXTRA $TT || echo "  ($ds FAILED)"
  fi
done
echo "done $(date)"

