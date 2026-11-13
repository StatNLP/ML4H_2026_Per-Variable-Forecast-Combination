#!/bin/bash
#SBATCH --job-name=stage1_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage1_%x_%j.out
#SBATCH --error=logs/stage1_%x_%j.err

# Parametric Stage-1 runner.
# Usage:  sbatch --job-name=<ds> jobs/run_stage1.sh <dataset>
#   <dataset> ∈ {mimic_iii, mimic_iv, hirid, eicu, umm}
#
# Run from the project root (so that the relative log paths above resolve).
# To choose a partition on the new cluster, pass it on the sbatch command line:
#     sbatch -p <partition> --job-name=mimic_iv jobs/run_stage1.sh mimic_iv

set -euo pipefail


if [[ $# -lt 1 ]]; then
    echo "ERROR: missing dataset name"; exit 2
fi
DATASET="$1"

# --chdir on the sbatch command sets the working directory before this runs.
WORKDIR="$(pwd)"
echo "WORKDIR=${WORKDIR}"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/data/long"

# Python binary: use the active conda env's `python` unless overridden.
PYTHON="${PYTHON:-python}"

OUTPUT_PATH="${WORKDIR}/data/long"
FEATURES_YAML="${WORKDIR}/configs/features.yaml"

# Per-dataset raw data path. Add new datasets here.
case "${DATASET}" in
    mimic_iii) DATA_PATH="${WORKDIR}/raw_data/mimic-iii-clinical-database-1.4" ;;
    mimic_iv)  DATA_PATH="${WORKDIR}/raw_data/physionet.org/files/mimiciv/3.1" ;;
    eicu)      DATA_PATH="${WORKDIR}/raw_data/eicu-collaborative-research-database-2.0" ;;
    hirid)     DATA_PATH="${WORKDIR}/raw_data/physionet.org/files/hirid/1.1.1" ;;
    umm)       DATA_PATH="${WORKDIR}/raw_data" ;;
    *) echo "ERROR: unknown dataset '${DATASET}'"; exit 3 ;;
esac

echo "Job started: $(date)"
echo "Node:        $(hostname)"
echo "Workdir:     ${WORKDIR}"
echo "Python:      $(${PYTHON} -V) [$(which ${PYTHON})]"
echo "Dataset:     ${DATASET}"
echo "Data path:   ${DATA_PATH}"
echo "Output:      ${OUTPUT_PATH}"

${PYTHON} - <<EOF
import sys
sys.path.insert(0, ".")
from harmonize.${DATASET} import run_pipeline

run_pipeline(
    data_path="${DATA_PATH}",
    output_path="${OUTPUT_PATH}",
    features_yaml="${FEATURES_YAML}",
)
EOF

echo "Job finished: $(date)"

sleep 5
MAXRSS_KB=$(sacct -j "${SLURM_JOB_ID:-0}" --noheader --format=MaxRSS --units=K 2>/dev/null \
    | grep -v "^$" | tail -1 | tr -d 'K ')
if [[ -n "${MAXRSS_KB}" && "${MAXRSS_KB}" -gt 0 ]]; then
    MAXRSS_GB=$(echo "scale=1; ${MAXRSS_KB} / 1048576" | bc)
    echo "Peak RAM (MaxRSS): ${MAXRSS_GB} GB  (${MAXRSS_KB} KB)"
fi
