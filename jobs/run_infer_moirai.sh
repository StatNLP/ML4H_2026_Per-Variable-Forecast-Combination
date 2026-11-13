#!/bin/bash
#SBATCH --job-name=infer_moirai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/moirai_%x_%j.out
#SBATCH --error=logs/moirai_%x_%j.err
#
# GPU wrapper for Moirai zero-shot inference (harmonize.infer_moirai). Runs in the
# SEPARATE py311_moirai env (uni2ts pins torch<2.5/numpy<2, incompatible with
# py313_tsf). Writes z-scored predictions to a .npy; the py313 eval consumes them
# via `eval_forecast --pred-file`. All CLI args forwarded.
set -euo pipefail
WORKDIR="$(pwd)"
PYTHON="${PYTHON_MOIRAI:-python}"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/data/fm_preds"

echo "Job started: $(date) on $(hostname)"
echo "Args: $*"
${PYTHON} -m harmonize.infer_moirai "$@"
echo "Job finished: $(date)"
