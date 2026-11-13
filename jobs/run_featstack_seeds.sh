#!/bin/bash
# jobs/run_featstack_seeds.sh — is the ensemble gain stable across patient splits?
#
# Every §27 number so far comes from ONE stay-grouped fit/eval split (seed 42). A gain of -6 %
# measured on a single partition of patients cannot be distinguished from a lucky split, which is
# the first thing a reviewer will ask. This re-runs the ladder over several split seeds and
# reports the spread; the combiners are closed-form numpy over cached predictions, so each seed
# costs minutes, not a GPU allocation.
#SBATCH --job-name=fs_seeds
#SBATCH --partition=cpu-single
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=logs/fs_seeds_%j.out
#SBATCH --error=logs/fs_seeds_%j.err
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"; cd "$ROOT"; export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PY="${PYTHON:-python}"
for ds in ${*:-umm hirid mimic_iii mimic_iv eicu}; do
  h5="data/dataset/${ds}_seed42_val__all_4h_fwd.h5"; [[ -f "$h5" ]] || continue
  PF=""
  for f in data/base_preds/*__${ds}_val.npy; do n=$(basename "$f" "__${ds}_val.npy"); PF="$PF ${n}=$f"; done
  for f in data/fm_preds/tirex2_${ds}_val.npy data/fm_preds/moirai_small_${ds}_val.npy; do
    [[ -f "$f" ]] && PF="$PF $(basename $f .npy | sed "s/_${ds}_val//")=$f"; done
  for seed in 42 0 7 123 2024; do
    echo "##### $ds split-seed $seed"
    $PY -m harmonize.ensemble_eval --mode featstack --test-h5 "$h5" --pred-files $PF \
        --n-clusters 1 --split-seed $seed 2>&1 \
      | grep -E "^ *${ds} +(simplex_var|greedy_var|inv_mse_var|rank_harmonic_top5|best_single) "
  done
done
echo "done $(date)"
