#!/bin/bash
# jobs/run_cfg_bundle.sh — run a GRID of configs for one (family, dataset) in ONE allocation.
#
# A 200-epoch run is ~25-33 min, so one allocation per config pays a queue wait per half hour of
# work. Bundling one dataset x one capacity (12 configs, ~6 h) keeps different (dataset, capacity)
# pairs parallel while cutting the number of queue waits by 12.
#
#   LAYERS="1 2 3" DROPOUTS="0.0 0.2" LRS="1e-4 3e-4" \
#     sbatch jobs/run_cfg_bundle.sh <family> <dataset> <d_model>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"; cd "$ROOT"
fam=$1; ds=$2; d=$3; ff="${DFF:-$((d*2))}"; SEED="${SEED:-123}"
# These were hardcoded and are wrong for anything but the formers/denseAR: STraTS uses 4 heads and
# batch 128. A schedule axis is also needed, and it MUST enter the run name — without it every
# schedule would write to the same directory and the skip-if-exists check would silently drop all
# but the first.
NHEADS="${NHEADS:-8}"; BS="${BS:-256}"; EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-6}"; PATLR="${PATLR:-3}"
SCHEDULE="${SCHEDULE:-}"; PFX="${RUNPREFIX:-hp}"
for L in ${LAYERS:?}; do
  for dr in ${DROPOUTS:?}; do
    for lr in ${LRS:?}; do
      tag="_dr${dr}_lr${lr}"      # always labelled; the old conditional left one cell nameless
      # Every axis that changes the MODEL must be in the name. Relying on the caller to encode
      # the schedule in RUNPREFIX is a trap: two schedules under one prefix silently collide and
      # the second bundle prints "skip" for every cell, yielding a "sweep" that is one schedule.
      stag=""; [[ -n "$SCHEDULE" && "$SCHEDULE" != "student" ]] && stag="_${SCHEDULE//_/}"
      ptag=""; [[ "$PATIENCE" != "6" ]] && ptag="_p${PATIENCE}"
      # d_ff is an AXIS (denseAR's own search swept 2x/3x/4x), so it must be in the name or the
      # ratios collide and the skip guard keeps only the first. STraTS references are all 4x.
      fftag=""; [[ "$ff" != "$((d*2))" ]] && fftag="_ff$((ff/d))x"
      run="${fam}_${PFX}_${ds}_d${d}_L${L}${tag}${stag}${ptag}${fftag}_s${SEED}"
      # Gate on best.pt, not the directory: train_forecast mkdirs BEFORE training, so a
      # crashed or walltime-killed cell left a directory and was skipped forever on
      # resubmit. archive_existing_run only archives when best.pt exists, so reruns are safe.
      [[ -f "checkpoints/$run/best.pt" ]] && { echo "skip $run (complete)"; continue; }
      echo "=== $run  $(date) ==="
      bash jobs/run_stage4.sh "$fam" "$ds" all_4h_fwd \
        --d-model "$d" --d-ff "$ff" --n-heads "$NHEADS" --n-layers "$L" --dropout "$dr" \
        --lr "$lr" --loss mse --epochs "$EPOCHS" --batch-size "$BS" \
        --patience-stop "$PATIENCE" --patience-lr "$PATLR" \
        ${SCHEDULE:+--schedule "$SCHEDULE"} \
        --seed "$SEED" --run-name "$run" || echo "  ($run failed)"
    done
  done
done
echo "bundle done $(date)"
