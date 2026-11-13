#!/bin/bash
#SBATCH --job-name=stage4_%x
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/stage4_%x_%j.out
#SBATCH --error=logs/stage4_%x_%j.err

# Parametric Stage-4 forecast trainer.
# Usage:
#   sbatch --job-name=<short> jobs/run_stage4.sh <model> <dataset> <feature-tag> [extra CLI args...]
#
#   <model>        ∈ {persistence, dlinear, dense_autoregressive}
#   <dataset>      ∈ {umm, mimic_iii, mimic_iv, hirid, eicu, all}
#   <feature-tag>  Stage-3 tag, e.g. sofa_first_fwd  (matches HDF5 suffix)
#   extra CLI args are forwarded verbatim to `harmonize.train_forecast`.
#
# GPU selection on helix (A40 / A100 / H100 / H200) — pick via partition OR gres.
# The CLI flag wins over the #SBATCH directive above.
#
#   # By partition (recommended on helix):
#   sbatch -p gpu_a40       --job-name=denseAR_umm jobs/run_stage4.sh dense_autoregressive umm sofa_first_fwd
#   sbatch -p gpu_a100_il   --job-name=denseAR_mim3 jobs/run_stage4.sh dense_autoregressive mimic_iii sofa_4h_fwd
#   sbatch -p gpu_h100      --job-name=denseAR_big  jobs/run_stage4.sh dense_autoregressive all sofa_4h_fwd
#   sbatch -p gpu_h200      --job-name=denseAR_big  jobs/run_stage4.sh dense_autoregressive all sofa_4h_fwd
#
#   # By gres (helix uses UPPERCASE type names):
#   sbatch --gres=gpu:A40:1   --job-name=...  jobs/run_stage4.sh ...
#   sbatch --gres=gpu:A100:1  --job-name=...  jobs/run_stage4.sh ...
#   sbatch --gres=gpu:H100:1  --job-name=...  jobs/run_stage4.sh ...
#
# Generic clusters that just expose `--gres=gpu:1` work with the directive above
# unchanged; pass `--constraint=<type>` if your cluster filters that way instead.

set -euo pipefail

if [[ $# -lt 3 ]]; then
    echo "ERROR: missing args. Usage:"
    echo "  sbatch --job-name=<short> $0 <model> <dataset> <feature-tag> [extra CLI args...]"
    exit 2
fi
MODEL="$1"; DATASET="$2"; TAG="$3"
shift 3
EXTRA_ARGS=("$@")

# The list used to be hardcoded here and went stale: it held 6 of the trainer's 16 models, so
# `strats` (and gru/lstm/mamba/patchtst/...) were rejected by the launcher despite being valid,
# failing in 1 s. Ask the trainer instead — one source of truth.
case "${MODEL}" in
    zeroforecast|persistence|dlinear|ets|arima|dense_autoregressive|dense_pteacher|dense_dms|\
    gru|lstm|mamba|informer|patchtst|autoformer|strats|strats_raw) ;;
    *) echo "ERROR: unknown model '${MODEL}'."; exit 3 ;;
esac

# NB: under sbatch, BASH_SOURCE points at SLURM's spooled copy of this script, so
# "$(dirname $BASH_SOURCE)/.." resolves to a system directory and the mkdir below fails with
# "Permission denied" 2 s in — which is how 20 queued re-runs died without training a step.
# Prefer the submit directory (SLURM sets it), then an explicit ROOT, then the source path.
WORKDIR="${SLURM_SUBMIT_DIR:-}"
if [[ -z "${WORKDIR}" || ! -d "${WORKDIR}/harmonize" ]]; then
    WORKDIR="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
fi
[[ -d "${WORKDIR}/harmonize" ]] || { echo "ERROR: bad WORKDIR ${WORKDIR}"; exit 2; }
cd "${WORKDIR}"
mkdir -p logs checkpoints

# Defaulting to bare `python` resolved to /usr/bin/python (no torch) whenever the caller did not
# export PYTHON — e.g. an `sbatch --wrap` that does not inherit a conda env. The CUDA preflight
# below then failed on `import torch`, could not tell that apart from "GPU busy", and requeued the
# job forever (job 14421490 reached Restarts=6 before this was found). Match every other script in
# notebooks/ and point at the env directly.
PYTHON="${PYTHON:-python}"

# --- CUDA preflight -------------------------------------------------------------------------
# Several jobs scheduled onto one node can each be granted --gres=gpu:1 yet collide on the same
# physical device (exclusive compute mode): the first process gets a context, the rest die with
# "CUDA-capable device(s) is/are busy or unavailable" seconds in. That happened to 4 of 5 TiRex-2
# jobs on o09c02. A 200-epoch training run is far too expensive to lose that way, so probe the
# device first and REQUEUE instead of burning the allocation.
# Only meaningful when a GPU was actually requested: on a CPU allocation cuda.is_available() is
# correctly False, and treating that as "device busy" requeues the job forever (the DLinear sweep
# hit Restarts=20 before this guard was added).
GPU_REQUESTED=0
[[ "${SLURM_JOB_GRES:-}" == *gpu* ]] && GPU_REQUESTED=1
[[ "${SLURM_GPUS_ON_NODE:-0}" -gt 0 ]] 2>/dev/null && GPU_REQUESTED=1
[[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "NoDevFiles" ]] && GPU_REQUESTED=1
# A broken interpreter must NOT look like a busy GPU: `import torch` failing is permanent, so
# requeueing on it loops until the job is noticed by hand. Probe the environment first and fail
# loudly, and only requeue for the transient condition (device present but unavailable).
if [[ "$GPU_REQUESTED" == "1" ]]; then
    if ! "${PYTHON}" -c "import torch" 2>/dev/null; then
        echo "FATAL: ${PYTHON} cannot import torch — this is an environment fault, not a busy GPU."
        echo "       Not requeueing (that would loop forever). Set PYTHON to a torch-capable env."
        exit 78
    fi
    if ! "${PYTHON}" -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.zeros(8,device='cuda').sum().item()==0 else 1)" 2>/dev/null; then
        # The cap belongs HERE, not before the probe: a job that has already accumulated restarts
        # from an earlier fault must still be allowed to run once the fault is fixed.
        if [[ "${SLURM_RESTART_COUNT:-0}" -ge 12 ]]; then
            echo "FATAL: ${SLURM_RESTART_COUNT} requeues and still no usable GPU — giving up."
            exit 79
        fi
        echo "CUDA busy on $(hostname) — requeueing job ${SLURM_JOB_ID:-?} (restart ${SLURM_RESTART_COUNT:-0})"
        if [[ -n "${SLURM_JOB_ID:-}" ]]; then scontrol requeue "${SLURM_JOB_ID}" && sleep 30 && exit 0; fi
        exit 75
    fi
fi
if [[ "$GPU_REQUESTED" == "1" ]]; then
    echo "CUDA preflight OK on $(hostname): $(${PYTHON} -c 'import torch;print(torch.cuda.get_device_name(0))')"
else
    echo "CPU allocation on $(hostname) — CUDA preflight skipped"
fi


DATA_DIR="${WORKDIR}/data/dataset"
TRAIN_H5="${DATA_DIR}/${DATASET}_seed42_train__${TAG}.h5"
VAL_H5="${DATA_DIR}/${DATASET}_seed42_val__${TAG}.h5"
RUN_NAME="${MODEL}_${DATASET}_${TAG}_${SLURM_JOB_ID:-local}"

if [[ ! -f "${TRAIN_H5}" ]]; then echo "ERROR: train HDF5 not found: ${TRAIN_H5}"; exit 4; fi
if [[ ! -f "${VAL_H5}"   ]]; then echo "ERROR: val   HDF5 not found: ${VAL_H5}";   exit 4; fi

echo "Job started: $(date)"
echo "Node:        $(hostname)"
echo "Workdir:     ${WORKDIR}"
echo "Python:      $(${PYTHON} -V) [$(which ${PYTHON})]"
echo "GPU:         $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'n/a')"
echo "Model:       ${MODEL}"
echo "Dataset:     ${DATASET}"
echo "Feature tag: ${TAG}"
echo "Train H5:    ${TRAIN_H5}"
echo "Val H5:      ${VAL_H5}"
echo "Run name:    ${RUN_NAME}"
echo "Extra args:  ${EXTRA_ARGS[*]:-<none>}"

${PYTHON} -m harmonize.train_forecast \
    --model "${MODEL}" \
    --train-h5 "${TRAIN_H5}" \
    --val-h5   "${VAL_H5}" \
    --output-dir "checkpoints" \
    --run-name "${RUN_NAME}" \
    "${EXTRA_ARGS[@]}"

echo "Job finished: $(date)"

sleep 5
MAXRSS_KB=$(sacct -j "${SLURM_JOB_ID:-0}" --noheader --format=MaxRSS --units=K 2>/dev/null \
    | grep -v "^$" | tail -1 | tr -d 'K ')
if [[ -n "${MAXRSS_KB}" && "${MAXRSS_KB}" -gt 0 ]]; then
    MAXRSS_GB=$(echo "scale=1; ${MAXRSS_KB} / 1048576" | bc)
    echo "Peak RAM (MaxRSS): ${MAXRSS_GB} GB  (${MAXRSS_KB} KB)"
fi
