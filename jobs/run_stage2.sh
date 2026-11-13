#!/bin/bash
#SBATCH --job-name=stage2_preproc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=logs/stage2_preproc_%j.out
#SBATCH --error=logs/stage2_preproc_%j.err

# Run from the project root. Override partition with `sbatch -p <name>`.

set -euo pipefail


WORKDIR="$(pwd)"
echo "WORKDIR=${WORKDIR}"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/data/splits"

PYTHON="${PYTHON:-python}"

echo "Job started: $(date)"
echo "Node:        $(hostname)"
echo "Workdir:     ${WORKDIR}"
echo "Python:      $(${PYTHON} -V) [$(which ${PYTHON})]"

# Datasets: pass as positional args to restrict (e.g. `run_stage2.sh hirid`);
# defaults to all five. Restricting is important when re-running one dataset
# while training jobs read the other datasets' splits (avoids overwriting them).
DATASETS=("$@")
[[ ${#DATASETS[@]} -eq 0 ]] && DATASETS=(eicu hirid mimic_iii mimic_iv umm)
echo "Datasets:    ${DATASETS[*]}"

${PYTHON} -m harmonize.stage2_preprocessing \
    --ds "${DATASETS[@]}" \
    --long-dir   data/long \
    --output-dir data/splits \
    --seed 42

echo "Job finished: $(date)"

sleep 5
MAXRSS_KB=$(sacct -j "${SLURM_JOB_ID:-0}" --noheader --format=MaxRSS --units=K 2>/dev/null \
    | grep -v "^$" | tail -1 | tr -d 'K ')
if [[ -n "${MAXRSS_KB}" && "${MAXRSS_KB}" -gt 0 ]]; then
    MAXRSS_GB=$(echo "scale=1; ${MAXRSS_KB} / 1048576" | bc)
    echo "Peak RAM (MaxRSS): ${MAXRSS_GB} GB  (${MAXRSS_KB} KB)"
fi
