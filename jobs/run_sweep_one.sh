#!/bin/bash
#SBATCH --output=logs/sweep_%x_%j.out
#SBATCH --error=logs/sweep_%x_%j.err
#
# One-combo wrapper invoked by `harmonize.hpsweep.submit`. The sbatch
# command (assembled in submit.py) supplies all SLURM resource flags
# (partition, gres, time, mem, cpus-per-task, job-name) on the CLI, so
# this file only carries log-path defaults.
#
# Positional arguments:
#   $1                — run name (used as --run-name and as job name)
#   $2 .. $N          — forwarded to harmonize.train_forecast verbatim
#
# Run-from-root is assumed (same convention as the other run_*.sh).

set -euo pipefail


if [[ $# -lt 1 ]]; then
    echo "ERROR: missing run-name + train_forecast args"; exit 2
fi
RUN_NAME="$1"
shift

# --chdir on the sbatch command sets the working directory before this runs.
WORKDIR="$(pwd)"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/checkpoints"

PYTHON="${PYTHON:-python}"

echo "Job started:  $(date)"
echo "Node:         $(hostname)"
echo "Workdir:      ${WORKDIR}"
echo "Python:       $(${PYTHON} -V) [$(which ${PYTHON})]"
echo "GPU:          $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Run name:     ${RUN_NAME}"
echo "Forwarded:    $*"

${PYTHON} -m harmonize.train_forecast \
    --output-dir "checkpoints" \
    --run-name "${RUN_NAME}" \
    "$@"

echo "Job finished: $(date)"

sleep 5
MAXRSS_KB=$(sacct -j "${SLURM_JOB_ID:-0}" --noheader --format=MaxRSS --units=K 2>/dev/null \
    | grep -v "^$" | tail -1 | tr -d 'K ')
if [[ -n "${MAXRSS_KB}" && "${MAXRSS_KB}" -gt 0 ]]; then
    MAXRSS_GB=$(echo "scale=1; ${MAXRSS_KB} / 1048576" | bc)
    echo "Peak RAM (MaxRSS): ${MAXRSS_GB} GB"
fi
