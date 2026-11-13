#!/bin/bash
# submit_dump_preds.sh — fan out the base-model prediction cache as ONE JOB PER (dataset, model).
#
# Why per-experiment rather than one bundled job: the dumps are independent, so the fan-out runs
# them in parallel and wall-clock collapses to the slowest single model instead of their sum.
# The three trivially-fast families (dlinear/null/persistence: >10k windows/s) are bundled into
# one job per dataset — a separate allocation for a job that finishes in seconds is pure overhead.
#
#   bash jobs/submit_dump_preds.sh                     # all 5 datasets
#   DATASETS="umm hirid" bash jobs/submit_dump_preds.sh
set -euo pipefail
WS="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
DATASETS="${DATASETS:-umm hirid mimic_iii mimic_iv eicu}"
SPLIT="${SPLIT:-val}"
SLOW="dense_autoregressive informer autoformer gru lstm mamba patchtst dense_dms strats strats_raw"
FAST="dlinear null persistence"
mkdir -p "$WS/logs" "$WS/data/base_preds"

n=0
for ds in $DATASETS; do
  [[ -f "$WS/data/dataset/${ds}_seed42_${SPLIT}__all_4h_fwd.h5" ]] || { echo "skip $ds"; continue; }
  case "$ds" in
    umm|hirid)      MEM=48G;  T=04:00:00 ;;
    mimic_iii)      MEM=96G;  T=08:00:00 ;;
    *)              MEM=160G; T=12:00:00 ;;
  esac
  for m in $SLOW; do
    sbatch --chdir="$WS" --parsable --partition=cpu-single --cpus-per-task=32 --mem=$MEM \
      --time=$T --job-name="dp_${ds}_${m}" \
      --output="$WS/logs/dp_${ds}_${m}_%j.out" \
      --error="$WS/logs/dp_${ds}_${m}_%j.err" \
      --export=ALL,MODELS="$m",SPLIT="$SPLIT" \
      jobs/run_dump_preds.sh "$ds" >/dev/null && n=$((n+1))
  done
  sbatch --chdir="$WS" --parsable --partition=cpu-single --cpus-per-task=16 --mem=$MEM \
    --time=02:00:00 --job-name="dp_${ds}_fast" \
    --output="$WS/logs/dp_${ds}_fast_%j.out" \
    --error="$WS/logs/dp_${ds}_fast_%j.err" \
    --export=ALL,MODELS="$FAST",SPLIT="$SPLIT" \
    jobs/run_dump_preds.sh "$ds" >/dev/null && n=$((n+1))
  echo "  queued $ds: 10 slow + 1 bundled-fast"
done
echo "submitted $n CPU dump jobs -> data/base_preds/"
