#!/bin/bash
# jobs/run_seed_bundle.sh — run several seeds of ONE (family, dataset) in ONE allocation.
#
# A 200-epoch run here takes ~25-33 min, so submitting each seed separately means paying a queue
# wait per 30 min of work. Bundling the seeds of one configuration into a single allocation cuts
# the number of queue waits by the number of seeds while keeping different (family, dataset)
# combinations parallel — the granularity that actually matters for wall-clock.
#
#   SEEDS="456 789" sbatch jobs/run_seed_bundle.sh <family> <dataset> <extra args...>
#
# Seeds come from $SEEDS, NOT positionally: passing the hyperparameters as one quoted "$extra"
# argument did not survive sbatch's argv handling, so $3 became "--d-model" and the remaining
# hyperparameters were consumed as seed numbers. Ten jobs were queued that way.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"; cd "$ROOT"
fam=$1; ds=$2; shift 2
extra=("$@")                       # everything else is hyperparameters
seeds=(${SEEDS:?SEEDS env var required, e.g. SEEDS="456 789 1337"})
for s in "${seeds[@]}"; do
  # RUNPREFIX distinguishes multi-seed rounds: hardcoding "ms2" meant a second round silently
  # skipped every run because the first round's directories already existed.
  run="${fam}_${RUNPREFIX:-ms2}_${ds}_s${s}"
  if [[ -f "checkpoints/$run/best.pt" ]]; then echo "skip $run (complete)"; continue; fi
  echo "=== $run  $(date) ==="
  # LR/EPOCHS/BS were hardcoded here, which silently overrode any per-dataset value passed in
  # "${extra[@]}" and made a schedule comparison impossible (the schedule under test would have
  # been run at the wrong optimiser settings). They are now defaults that "${extra[@]}" can
  # override, because argparse takes the LAST occurrence of a repeated flag.
  bash jobs/run_stage4.sh "$fam" "$ds" all_4h_fwd \
      --lr "${LR:-3e-4}" --loss "${LOSS:-mse}" --epochs "${EPOCHS:-200}" \
      --batch-size "${BS:-256}" ${SCHEDULE:+--schedule "$SCHEDULE"} \
      "${extra[@]}" --seed "$s" --run-name "$run" \
      || echo "  ($run failed)"
done
echo "bundle done $(date)"
