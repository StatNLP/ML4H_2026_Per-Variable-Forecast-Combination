#!/bin/bash
# submit_eval_tirex2.sh
#
# Zero-shot TiRex-2 (NX-AI, arXiv:2607.01204) across every dataset — the RECENCY
# control for the FM ladder (§12.7): Chronos/Moirai/MIRA are 2024/25 vintage, TiRex-2
# is the newest natively-multivariate zero-shot FM (xLSTM backbone, 38.4M+44.1M).
#
# Same two-step dependency chain as Moirai, because tirex-2 pins xlstm~=2.0.3
# (torch 2.9.1 / numpy 2.1) and cannot share py313_tsf:
#   (1) infer  — py313_tirex writes data/fm_preds/tirex2_<ds>.npy            (GPU)
#   (2) eval   — py313 reads it via --pred-file → docs/tables/eval_tirex2_<ds>.json
# so TiRex-2 lands in the same table as Chronos-2 and Moirai (all native multivariate).
#
# Inference is inlined via --wrap rather than given its own run_infer_*.sh wrapper:
# it is a single module call and needs no extra plumbing.
#
# Usage:  bash jobs/submit_eval_tirex2.sh
#         DATASETS="umm" LIMIT="--limit 256" bash jobs/submit_eval_tirex2.sh   # smoke test
# After:  python 10_scripts/aggregate_eval.py
set -euo pipefail
WS="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
PY_TIREX="${PY_TIREX:-python}"
DATASETS="${DATASETS:-mimic_iii mimic_iv eicu hirid umm}"
# SPLIT: which H5 to score. `test` is the paper's held-out set; `val` is what the per-(variable,
# hour) stacking ensemble (§13.6b) fits on, so the FMs need val predictions to join that stack.
SPLIT="${SPLIT:-test}"
# INFER_ONLY=1 skips the eval chain — featstack consumes the raw .npy, not eval_<run>.json.
INFER_ONLY="${INFER_ONLY:-0}"
BS="${BS:-64}"
# xlstm's sLSTM cell resolves CUDA include paths AT IMPORT and dies with
# "CUDA_HOME environment variable is not set" if no CUDA module is loaded — the
# login node has a fallback, the compute nodes do not (job 14092368 failed at 29 s).
# 12.8 matches the torch 2.9.1+cu128 wheel in py313_tirex.
# flashrnn then JIT-compiles its CUDA kernels on the FIRST forward and needs `ninja`
# on PATH (job 14099999 died there at 62 s). ninja ships inside py313_tirex/bin, which
# is not on PATH because python is invoked by absolute path — so prepend the env bin.
# The compile is cached in ~/.cache/torch_extensions, hence: run ONE job cold, then the
# rest reuse it (launching all five from cold would compile five times and can race).
# ...and that compile needs GCC >= 9 for the host .cc file, while the node default is
# GCC 8.5 (job 14100006: 6/8 CUDA objects built, then flashrnn.cc hit
# "too old version of GCC"). 13.3 is within CUDA 12.8's supported compiler range.
CUDA_MOD="${CUDA_MOD:-12.8}"
GNU_MOD="${GNU_MOD:-13.3}"
# Measured warm throughput (job 14100013, A100, bs=64): ~13.7 windows/s, so
# umm 0.4h · hirid 0.5h · mimic_iii 1.8h · mimic_iv 2.8h · eicu 4.3h. eICU alone
# exceeds a 4h wall, so the default carries headroom for the largest test set.
INFER_TIME="${INFER_TIME:-10:00:00}"
LIMIT="${LIMIT:-}"
SUFFIX="${SUFFIX:-}"          # e.g. _smoke, to avoid clobbering a real run
mkdir -p "$WS/logs" "$WS/docs/tables" "$WS/data/fm_preds"

n=0
for ds in $DATASETS; do
  test_h5="data/dataset/${ds}_seed42_${SPLIT}__all_4h_fwd.h5"
  [[ -f "$WS/$test_h5" ]] || { echo "skip $ds: no $test_h5"; continue; }
  long="data/long/${ds}_long.parquet"; long_arg=""
  [[ -f "$WS/$long" ]] && long_arg="--long-parquet $long"
  sfx=""; [[ "$SPLIT" != "test" ]] && sfx="_${SPLIT}"
  npy="data/fm_preds/tirex2_${ds}${sfx}${SUFFIX}.npy"
  run="tirex2_${ds}${sfx}${SUFFIX}"

  jid=$(sbatch --chdir="$WS" --parsable \
      --partition=gpu-single --gres=gpu:1 --cpus-per-task=4 --mem=64G \
      --time=${INFER_TIME} --job-name="infer_${run}" \
      \
      --output="$WS/logs/tirex2_infer_${ds}${sfx}${SUFFIX}_%j.out" \
      --error="$WS/logs/tirex2_infer_${ds}${sfx}${SUFFIX}_%j.err" \
      --wrap "              module load compiler/gnu/${GNU_MOD}; \
              module load devel/cuda/${CUDA_MOD}; \
              export PATH=\"$(dirname $PY_TIREX):\$PATH\"; \
              export HF_HUB_OFFLINE=1; \
              echo \"CUDA_HOME=\$CUDA_HOME ninja=\$(command -v ninja) gcc=\$(gcc -dumpversion)\"; \
              $PY_TIREX -c \"import torch,sys; sys.exit(0 if torch.cuda.is_available() and torch.zeros(8,device='cuda').sum().item()==0 else 1)\" || \
                { echo \"CUDA busy on \$(hostname) — requeueing\"; scontrol requeue \$SLURM_JOB_ID; sleep 30; exit 0; }; \
              $PY_TIREX -m harmonize.infer_tirex2 \
                  --test-h5 '$test_h5' --out '$npy' --batch-size $BS $LIMIT")

  if [[ "$INFER_ONLY" == "1" ]]; then
    echo "  queued tirex2 ${ds}${sfx}  (infer $jid, no eval)"; n=$((n+1)); continue
  fi
  sbatch --chdir="$WS" --parsable --dependency=afterok:$jid \
      --partition=gpu-single --gres=gpu:1 --cpus-per-task=4 --mem=64G \
      --time=01:30:00 --job-name="eval_${run}" \
      \
      --output="$WS/logs/eval_${run}_%j.out" \
      --error="$WS/logs/eval_${run}_%j.err" \
      jobs/run_eval.sh \
          --pred-file "$npy" --test-h5 "$test_h5" $long_arg \
          --out "docs/tables/eval_${run}.json" \
      >/dev/null && { echo "  queued tirex2 ${ds}${SUFFIX}  (infer $jid → eval)"; n=$((n+1)); }
done
echo ""
echo "Queued $n TiRex-2 infer+eval chains (datasets: $DATASETS, bs=$BS). After:"
echo "  python 10_scripts/aggregate_eval.py"
