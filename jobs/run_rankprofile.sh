#!/bin/bash
# Rank-profile ladder (equal-k / harm-k / fit-k, k in {3,5,14}) for one database.
# MODE=dev : fit on half of val, score the held-out half.  MODE=test: fit on val, score test.
# Weights always come from validation; test is only scored, never fitted on.
#SBATCH --partition=devel
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=logs/rp2_%x_%j.out
#SBATCH --error=logs/rp2_%x_%j.err
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"; cd "$ROOT"; export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
PY="${PYTHON:-python}"
ds="${DS:?set DS}"; mode="${MODE:?set MODE}"
vh="data/dataset/${ds}_seed42_val__all_4h_fwd.h5"; th="data/dataset/${ds}_seed42_test__all_4h_fwd.h5"
NAMES=$(ls data/base_preds/*__${ds}_val.npy | sed "s|.*/||;s|__${ds}_val.npy||" | sort)
VP=""; TP=""
for n in $NAMES; do
  [[ -f "data/base_preds/${n}__${ds}_test.npy" ]] || { echo "  !! $n missing on test"; exit 1; }
  VP="$VP ${n}=data/base_preds/${n}__${ds}_val.npy"; TP="$TP ${n}=data/base_preds/${n}__${ds}_test.npy"
done
for fm in tirex2 moirai_small; do
  v="data/fm_preds/${fm}_${ds}_val.npy"; t="data/fm_preds/${fm}_${ds}.npy"
  [[ -f "$v" && -f "$t" ]] && { VP="$VP ${fm}=$v"; TP="$TP ${fm}=$t"; }
done
echo "=== ${mode^^} $ds  $(date) ==="
if [[ "$mode" == "dev" ]]; then
  $PY -m harmonize.ensemble_eval --mode featstack --test-h5 "$vh" --pred-files $VP --n-clusters 1
else
  $PY -m harmonize.ensemble_eval --mode featstack --test-h5 "$vh" --pred-files $VP \
      --eval-h5 "$th" --eval-pred-files $TP --n-clusters 1
fi
echo "done $(date)"
