#!/bin/bash
#SBATCH --job-name=mk_triplets
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/mk_triplets_%j.out
#SBATCH --error=logs/mk_triplets_%j.out
set -euo pipefail
cd "${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
export PYTHONPATH=${PROJECT_ROOT:-$(pwd)}:${PYTHONPATH:-}
PY="${PYTHON:-python}"
$PY -m harmonize.stage3_make_triplets --dense-tag all_4h_fwd --seed 42 --datasets "$@"
