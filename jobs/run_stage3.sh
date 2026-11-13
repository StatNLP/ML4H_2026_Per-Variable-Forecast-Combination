#!/bin/bash
#SBATCH --job-name=stage3_dataset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3_dataset_%j.out
#SBATCH --error=logs/stage3_dataset_%j.err

# Run from the project root. Override partition with `sbatch -p <name>`.
# All Stage-3 CLI args are forwarded: pass `--features`, `--stride`, `--imputation`,
# `--predict-interventions`, `--datasets`, `--window-in`, `--window-out`.
#
# Examples:
#   sbatch jobs/run_stage3.sh --features sofa --stride first --imputation fwd --datasets umm
#   sbatch jobs/run_stage3.sh --features covered10 --stride 4h --imputation fwd
#     (uses default --datasets = all five)

set -euo pipefail


WORKDIR="$(pwd)"
echo "WORKDIR=${WORKDIR}"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/data/dataset"

PYTHON="${PYTHON:-python}"

echo "Job started: $(date)"
echo "Node:        $(hostname)"
echo "Workdir:     ${WORKDIR}"
echo "Python:      $(${PYTHON} -V) [$(which ${PYTHON})]"
echo "Extra args:  $*"

${PYTHON} -m harmonize.stage3_make_dataset \
    --splits-dir data/splits \
    --output-dir data/dataset \
    --seed 42 \
    "$@"

echo "Job finished: $(date)"

sleep 5
MAXRSS_KB=$(sacct -j "${SLURM_JOB_ID:-0}" --noheader --format=MaxRSS --units=K 2>/dev/null \
    | grep -v "^$" | tail -1 | tr -d 'K ')
if [[ -n "${MAXRSS_KB}" && "${MAXRSS_KB}" -gt 0 ]]; then
    MAXRSS_GB=$(echo "scale=1; ${MAXRSS_KB} / 1048576" | bc)
    echo "Peak RAM (MaxRSS): ${MAXRSS_GB} GB"
fi
