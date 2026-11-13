#!/bin/bash
# jobs/run_train_gpu.sh
# SLURM wrapper for a single train_forecast run.
#
# All train_forecast.py CLI args are forwarded directly.
# Request GPU via:  sbatch --partition=gpu-single --gres=gpu:1 jobs/run_train_gpu.sh <args>
#
# Example:
#   sbatch --partition=gpu-single --gres=gpu:1 --time=16:00:00 \
#     jobs/run_train_gpu.sh \
#       --model dense_autoregressive \
#       --train-h5 data/dataset/mimic_iii_seed42_train__all_first_fwd.h5 \
#       --val-h5   data/dataset/mimic_iii_seed42_val__all_first_fwd.h5 \
#       --run-name mimic_iii__d128_l2_dr0.1_s42

#SBATCH --job-name=train_forecast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

set -euo pipefail


WORKDIR="$(pwd)"
PYTHON="${PYTHON:-python}"

echo "Job started:  $(date)"
echo "Node:         $(hostname)"
echo "Workdir:      ${WORKDIR}"
echo "SLURM job:    ${SLURM_JOB_ID:-local}"
echo "Args:         $*"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
fi

mkdir -p "${WORKDIR}/logs" "${WORKDIR}/checkpoints"

${PYTHON} -m harmonize.train_forecast "$@"

echo "Job finished: $(date)"
sleep 5
MAXRSS_KB=$(sacct -j "${SLURM_JOB_ID:-0}" --noheader --format=MaxRSS --units=K 2>/dev/null \
    | grep -v "^$" | tail -1 | tr -d 'K ' || echo "")
if [[ -n "${MAXRSS_KB}" && "${MAXRSS_KB}" -gt 0 ]]; then
    echo "Peak RAM: $(echo "scale=1; ${MAXRSS_KB}/1048576" | bc) GB"
fi
