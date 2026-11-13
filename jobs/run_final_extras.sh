#!/bin/bash
# Appendix artefacts for the FINAL (val-fit -> test-score) run: paired randomisation test,
# win shares on ALL five databases, per-model x per-system tables, per-horizon gain.
# Weights come from validation; nothing is selected on test, so this does not re-spend it.
#SBATCH --job-name=final_extras
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=12:00:00
#SBATCH --output=logs/final_extras_%j.out
#SBATCH --error=logs/final_extras_%j.err
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"; cd "$ROOT"; export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PY="${PYTHON:-python}"
NPERM="${NPERM:-10000}"
# CLINICAL=1: SOFA/SAPS-II from the test forecasts (final_clinical) instead of the appendix tables.
XFLAG="--final-extras --n-perm $NPERM"; [[ "${CLINICAL:-0}" == "1" ]] && XFLAG="--final-clinical"; [[ "${SUPPORT:-0}" == "1" ]] && XFLAG="--final-support"
XFLAG=${XFLAGS:-$XFLAG}
# Rebuttal test pools (declared before running, scored once): NOFM=1 drops Moirai/TiRex-2, ADD
# appends members (arima ets lightgbmfix). Such a run MUST carry RBTAG; its tables then go to
# docs/tables/final_rb_<tag>/ and never touch the published FINAL tables.
if [[ "${NOFM:-0}" == "1" || -n "${ADD:-}" ]]; then
  [[ -n "${RBTAG:-}" ]] || { echo "NOFM/ADD need RBTAG"; exit 1; }
fi
TT=""; [[ -n "${RBTAG:-}" ]] && TT="--table-tag _rb_${RBTAG}"
for ds in ${*:-umm hirid mimic_iii mimic_iv eicu}; do
  vh="data/dataset/${ds}_seed42_val__all_4h_fwd.h5"; th="data/dataset/${ds}_seed42_test__all_4h_fwd.h5"
  [[ -f "$vh" && -f "$th" ]] || { echo "skip $ds"; continue; }
  # Pool = the manifest, never a directory glob (see run_final_test.sh for the incident).
  NAMES=$(awk -F'\t' -v d="$ds" '$1==d{print $2}' configs/pool_manifest.tsv | sort)
  [[ -n "$NAMES" ]] || { echo "  !! $ds missing from configs/pool_manifest.tsv"; continue; }
  for a in ${ADD:-}; do case $a in arima|ets) NAMES="$NAMES ${a}_${ds}_4h";; *) NAMES="$NAMES ${a}_${ds}";; esac; done
  VP=""; TP=""
  for n in $NAMES; do
    [[ -f "data/base_preds/${n}__${ds}_test.npy" ]] || { echo "  !! $n missing on test"; VP=""; break; }
    pv=data/base_preds/${n}__${ds}_val.npy; pt=data/base_preds/${n}__${ds}_test.npy
    [[ -f data/base_preds_protected/${n}__${ds}_val.npy ]]  && pv=data/base_preds_protected/${n}__${ds}_val.npy
    [[ -f data/base_preds_protected/${n}__${ds}_test.npy ]] && pt=data/base_preds_protected/${n}__${ds}_test.npy
    VP="$VP ${n}=$pv"; TP="$TP ${n}=$pt"
  done
  [[ -z "$VP" ]] && continue
  [[ "${NOFM:-0}" == "1" ]] || for fm in tirex2 moirai_small; do
    v="data/fm_preds/${fm}_${ds}_val.npy"; t="data/fm_preds/${fm}_${ds}.npy"
    [[ -f "$v" && -f "$t" ]] && { VP="$VP ${fm}=$v"; TP="$TP ${fm}=$t"; }
  done
  echo "=== $ds ${RBTAG:-published} ($(echo $VP | wc -w) members) $(date) ==="
  $PY -m harmonize.ensemble_eval --mode featstack \
      --test-h5 "$vh" --pred-files $VP --eval-h5 "$th" --eval-pred-files $TP \
      --n-clusters 1 $XFLAG $TT || echo "  ($ds FAILED)"
done
echo "done $(date)"
