#!/bin/bash
# jobs/run_dump_preds.sh — CPU job: cache every base model's val predictions to .npy.
#
# Inference is the only GPU-shaped step in the §27 stacking study; the combiners themselves are
# closed-form numpy over sufficient statistics. Caching predictions once turns every later
# experiment (new combiner, cluster count, ridge lambda, pool subset) into a seconds-long CPU
# job with no checkpoints and no GPU — which matters when the GPU queue is saturated.
#
#   sbatch jobs/run_dump_preds.sh umm hirid
#SBATCH --job-name=dump_preds
#SBATCH --partition=cpu-single
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dump_preds_%j.out
#SBATCH --error=logs/dump_preds_%j.err
set -uo pipefail
ROOT="${DISS_ROOT:-${PROJECT_ROOT:-$(pwd)}}"
cd "$ROOT"; export PYTHONPATH="$ROOT"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
PY="${PYTHON:-python}"
MODELS="${MODELS:-dense_autoregressive informer dlinear gru dense_dms lstm mamba patchtst autoformer strats strats_raw null persistence}"
SPLIT="${SPLIT:-val}"
# The default best-per-family dump must not run when a special mode is requested. It used to run
# first regardless and was skipped only because "${@:-umm hirid}" quotes the default list into ONE
# word ("umm hirid"), which matches no dataset. Had the quoting been right, every SEEDS/CLASSIC job
# would first have re-dumped the "best" checkpoint per family into the pool directory -- and with
# newer checkpoints on disk, "best" need not be the published pool member any more.
if [[ "${SEEDS:-0}" == "1" || "${CLASSIC:-0}" == "1" || "${LGBM:-0}" == "1" || "${LGBM_SWEEP:-0}" == "1" ]]; then
  :   # handled by the mode blocks below
else
for ds in ${@:-umm hirid}; do
    h5="data/dataset/${ds}_seed42_${SPLIT}__all_4h_fwd.h5"
    [[ -f "$h5" ]] || { echo "skip $ds (no $h5)"; continue; }
    CK=$("$PY" - "$ds" "$MODELS" <<'PYEOF'
import json, glob, sys
from pathlib import Path
ds, models = sys.argv[1], sys.argv[2].split()
best = {}
for jf in glob.glob("checkpoints/*/final_metrics.json"):
    try: j = json.loads(Path(jf).read_text())
    except Exception: continue
    c = j.get("combo", {}) or {}
    if f"{ds}_seed42_train__all_4h_fwd.h5" not in str(c.get("train_h5", "")): continue
    fam = str(c.get("model", ""))
    if fam not in models: continue
    mse = (j.get("val", {}) or {}).get("mse_per_var")
    if mse is None: continue
    d = str(Path(jf).parent)
    # Autoformer/Informer: only checkpoints trained AFTER the dead-decoder fix are usable. Their
    # pre-fix weights were trained under the broken architecture, so loading them into the
    # corrected code runs them in a configuration they never saw (measured: Informer 29 % worse
    # than its own trained capability), and the two eras' val metrics are not comparable.
    # Select on the TRAINING TIMESTAMP, not the run name: the matched-budget sweep runs are named
    # "*_hp_*" and are post-fix, so a name filter on "_fix_" would have silently discarded the
    # best available checkpoints and kept worse ones.
    if fam in ("autoformer", "informer"):
        cfgp = Path(d) / "config.json"
        try:
            started = json.loads(cfgp.read_text()).get("started_at", "")
        except Exception:
            started = ""
        if started < "2026-08-13T12:01":      # the decoder fix (commit 49a2a00)
            continue
    if fam not in best or mse < best[fam][0]: best[fam] = (float(mse), d)
print(" ".join(d for _, d in best.values()))
PYEOF
)
    echo "=== $ds: $(echo $CK | wc -w) models -> data/base_preds  $(date)"
    "$PY" -m harmonize.ensemble_eval --mode featstack --test-h5 "$h5" \
        --checkpoints $CK --dump-preds data/base_preds --batch-size 256 \
        || echo "  ($ds dump failed)"
done
echo "done $(date)"
fi

# SEEDS=1 — dump EVERY seed checkpoint of the strongest two families, not just the best one.
# The default selection above keeps one checkpoint per family (lowest val MSE), which is right for
# the 14-model pool but makes SbLR's variance control impossible: that control needs N seeds of ONE
# architecture as its own pool. Inference only; no retraining.
#   SEEDS=1 sbatch jobs/run_dump_preds.sh umm hirid mimic_iii mimic_iv eicu
if [[ "${SEEDS:-0}" == "1" ]]; then
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name=rb_dumpseeds 2>/dev/null || true
  for ds in ${*:-umm hirid mimic_iii mimic_iv eicu}; do
    for split in val test; do
      h5="data/dataset/${ds}_seed42_${split}__all_4h_fwd.h5"
      [[ -f "$h5" ]] || { echo "skip $ds/$split"; continue; }
      CK=$(ls -d checkpoints/strats_${ds}_ms_s* checkpoints/dense_autoregressive_${ds}_ms_s* 2>/dev/null \
           | while read d; do [[ -f "$d/best.pt" ]] && echo "$d"; done | tr '\n' ' ')
      [[ -n "$CK" ]] || { echo "skip $ds: no seed checkpoints"; continue; }
      echo "=== SEEDS $ds/$split: $(echo $CK|wc -w) checkpoints  $(date)"
      "$PY" -m harmonize.ensemble_eval --mode featstack --test-h5 "$h5" \
          --checkpoints $CK --dump-preds data/base_preds --batch-size 256 \
          || echo "  ($ds/$split dump failed)"
    done
  done
  echo "seed dumps done $(date)"; exit 0
fi

# CLASSIC=1 — dump ARIMA and ETS predictions (Paper 1's *_4h configs) on val and test, so they can
# join the combiner pool as a 16-model robustness check. JFiH: the pool has one linear model and no
# classical forecasters. Pool membership is the manifest, so these files do NOT enter the published
# 14-model pool by sitting in data/base_preds. Run as an array, one database per task:
#   CLASSIC=1 sbatch --array=0-4 jobs/run_dump_preds.sh
if [[ "${CLASSIC:-0}" == "1" ]]; then
  ALL=(umm hirid mimic_iii mimic_iv eicu)
  DS=${ALL[${SLURM_ARRAY_TASK_ID:-0}]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}" Name="rb_classic" 2>/dev/null || true
  CK="checkpoints/arima_${DS}_4h checkpoints/ets_${DS}_4h"
  for split in val test; do
    h5="data/dataset/${DS}_seed42_${split}__all_4h_fwd.h5"
    [[ -f "$h5" ]] || { echo "skip $DS/$split"; continue; }
    echo "=== CLASSIC $DS/$split  $(date)"
    "$PY" -m harmonize.ensemble_eval --mode featstack --test-h5 "$h5" \
        --checkpoints $CK --dump-preds data/base_preds --batch-size 512 \
        || { echo "  ($DS/$split dump FAILED)"; exit 1; }
  done
  echo "classic dumps done $DS $(date)"; exit 0
fi

# LGBM_SWEEP=1 — hyper-parameter sweep for LightGBM, ONE database (DS=...), development protocol.
# Every arm uses --fixed-rounds, so validation is never read during training; configurations are
# then compared by scoring the val predictions, which spends val exactly once, for selection.
# Test is untouched and stays available for the single final number.
#   DS=mimic_iii LGBM_SWEEP=1 sbatch --array=0-14 jobs/run_dump_preds.sh
if [[ "${LGBM_SWEEP:-0}" == "1" ]]; then
  DS="${DS:?set DS}"
  # Coordinate sweep around the published default (arm 0), one axis at a time so each result is
  # attributable. leaves/min_data = capacity vs regularisation; ff/l2 = decorrelation vs shrinkage;
  # lr x rounds = the usual trade; rounds alone = is 200 simply too few?
  ARMS=(
    "--fixed-rounds 200"
    "--fixed-rounds 200 --num-leaves 31"
    "--fixed-rounds 200 --num-leaves 127"
    "--fixed-rounds 200 --num-leaves 255"
    "--fixed-rounds 200 --min-data-in-leaf 20"
    "--fixed-rounds 200 --min-data-in-leaf 100"
    "--fixed-rounds 200 --min-data-in-leaf 200"
    "--fixed-rounds 200 --feature-fraction 0.3"
    "--fixed-rounds 200 --feature-fraction 0.8"
    "--fixed-rounds 200 --lambda-l2 0.1"
    "--fixed-rounds 200 --lambda-l2 10"
    "--fixed-rounds 500 --learning-rate 0.02"
    "--fixed-rounds 100 --learning-rate 0.1"
    "--fixed-rounds 400"
    "--fixed-rounds 800"
  )
  i=${SLURM_ARRAY_TASK_ID:-0}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${i}" Name="lgbm_sw" 2>/dev/null || true
  echo "=== LGBM sweep arm $i on $DS: ${ARMS[$i]}  $(date)"
  "$PY" -m harmonize.models.lightgbm_fc --dataset "$DS" --threads "${SLURM_CPUS_PER_TASK:-32}" \
      ${ARMS[$i]} --tag-suffix "_sw${i}" \
      || { echo "  (arm $i FAILED)"; exit 1; }
  echo "sweep arm $i done $(date)"; exit 0
fi

# LGBM=1 — train and dump a LightGBM member (harmonize/models/lightgbm_fc.py): fit on train, early
# stopping on val, predictions written for val and test. JFiH: no tree ensembles in the pool. Like
# ARIMA/ETS it is a robustness member under the development protocol, not part of the 14-model pool.
#   LGBM=1 sbatch --array=0-4 jobs/run_dump_preds.sh
if [[ "${LGBM:-0}" == "1" ]]; then
  ALL=(umm hirid mimic_iii mimic_iv eicu)
  DS=${ALL[${SLURM_ARRAY_TASK_ID:-0}]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}" Name="rb_lgbm" 2>/dev/null || true
  echo "=== LGBM $DS  $(date)"
  "$PY" -m harmonize.models.lightgbm_fc --dataset "$DS" --threads "${SLURM_CPUS_PER_TASK:-32}" ${LGBM_ARGS:-} \
      || { echo "  ($DS LightGBM FAILED)"; exit 1; }
  echo "lgbm done $DS $(date)"; exit 0
fi

# ---------------------------------------------------------------------------------------------
# FIRSTDAY=1 — dump the MANIFEST pool's predictions on ADMISSION WINDOWS ONLY (t0_hour == 0),
# for train / val / test. This is what the §16.10a outcome heads consume: they run
# --first-day-only, so a full-split dump would materialise ~23x more data than they can use
# (eICU train is 6.2 GB per member at full width, ~270 MB at admission width).
#
# The pool is pinned to configs/pool_manifest.tsv — the published 14-member pool — not to whatever
# happens to be in checkpoints/, so the combined trajectory this feeds is the published ENSEMBLE.
# Files land as <name>__<ds>_<split>_fd.npy.
#
#   FIRSTDAY=1 sbatch --array=0-4 jobs/run_dump_preds.sh
if [[ "${FIRSTDAY:-0}" == "1" ]]; then
  ALL=(umm hirid mimic_iii mimic_iv eicu)
  ds=${ALL[${SLURM_ARRAY_TASK_ID:-0}]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name="fd_$ds" 2>/dev/null || true
  CK=""
  while read -r d name; do
    [[ "$d" == "$ds" ]] || continue
    [[ -d "checkpoints/$name" ]] && CK="$CK checkpoints/$name"
  done < configs/pool_manifest.tsv
  [[ -z "$CK" ]] && { echo "no manifest checkpoints for $ds"; exit 1; }
  echo "=== $ds: $(echo $CK | wc -w) manifest models, admission windows, 3 splits  $(date)"
  for split in train val test; do
    h5="data/dataset/${ds}_seed42_${split}__all_4h_fwd.h5"
    [[ -f "$h5" ]] || { echo "  missing $h5"; continue; }
    echo "--- $ds/$split $(date +%H:%M)"
    "$PY" -m harmonize.ensemble_eval --mode featstack --test-h5 "$h5" \
        --checkpoints $CK --dump-preds data/base_preds --rows first-day --batch-size 256 \
        || echo "  ($ds/$split dump failed)"
  done
  echo "done $ds $(date)"
  exit 0
fi

# ---------------------------------------------------------------------------------------------
# FMTRAJ=1 — dump FOUNDATION-MODEL forecast trajectories on admission windows, for the §16.10a
# trajectory ladder. The FMs are the ladder's CLEAN rungs: they are zero-shot, so unlike denseAR,
# LightGBM and the ensemble their train-split predictions are NOT in-sample, and a dose-response
# in forecast quality has to hold across them too, not only across the fitted models.
#
# Three FMs spanning FM quality: bolt-tiny (weakest), bolt-base (the ladder's best Chronos),
# chronos-2 (multivariate). Array index = db*9 + split*3 + fm, so each task is one
# (database, split, FM) cell — the longest is eICU/train at ~22 h for one FM.
#
#   FMTRAJ=1 sbatch --array=0-8   jobs/run_dump_preds.sh   # umm
#   FMTRAJ=1 sbatch --array=9-17  jobs/run_dump_preds.sh   # hirid
#   FMTRAJ=1 sbatch --array=18-44 jobs/run_dump_preds.sh   # the three large ones
if [[ "${FMTRAJ:-0}" == "1" ]]; then
  DBS=(umm hirid mimic_iii mimic_iv eicu)
  SPLITS=(train val test)
  FMS=("chronos:amazon/chronos-bolt-tiny" "chronos:amazon/chronos-bolt-base" "chronos2:amazon/chronos-2")
  i=${SLURM_ARRAY_TASK_ID:-0}
  ds=${DBS[$((i / 9))]}; split=${SPLITS[$(((i / 3) % 3))]}; fm=${FMS[$((i % 3))]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name="fmt_${ds}_${split}" 2>/dev/null || true
  h5="data/dataset/${ds}_seed42_${split}__all_4h_fwd.h5"
  [[ -f "$h5" ]] || { echo "missing $h5"; exit 1; }
  echo "=== FM trajectory: $ds/$split  $fm  $(date)"
  "$PY" -m harmonize.eval_fm_suite --test-h5 "$h5" --models "$fm" \
      --dump-preds data/base_preds --rows first-day --batch-size 256 \
      || { echo "  ($ds/$split $fm FAILED)"; exit 1; }
  echo "done $ds/$split $(date)"; exit 0
fi

# ---------------------------------------------------------------------------------------------
# ENSTRAJ=1 — materialise the ENSEMBLE's forecast trajectory on admission windows, for the
# §16.10a trajectory ladder's top rung.
#
# The weights are fitted ONCE on the TRAIN split's fit half and then reused unchanged for val and
# test, so the ensemble is one fixed function rather than a per-split refit. Fitting on train (not
# val) also keeps the combiner's fitting data disjoint from the split the downstream head selects
# on.
#
#   ENSTRAJ=1 sbatch --array=0-4 jobs/run_dump_preds.sh
if [[ "${ENSTRAJ:-0}" == "1" ]]; then
  ALL=(umm hirid mimic_iii mimic_iv eicu)
  ds=${ALL[${SLURM_ARRAY_TASK_ID:-0}]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name="enstraj_$ds" 2>/dev/null || true
  WSET="${WSET:-simplex_shrunk500_var}"
  OUT=data/base_preds
  WNPZ="$OUT/weights_${WSET}__${ds}.npz"
  for split in train val test; do
    h5="data/dataset/${ds}_seed42_${split}__all_4h_fwd.h5"
    # the pool: the manifest members' admission-window dumps for this split
    PF=""
    while read -r d name; do
      [[ "$d" == "$ds" ]] || continue
      f="$OUT/${name}__${ds}_${split}_fd.npy"; [[ -f "$f" ]] && PF="$PF ${name}=$f"
    done < configs/pool_manifest.tsv
    n=$(echo $PF | wc -w)
    [[ $n -lt 2 ]] && { echo "  $ds/$split: only $n members, skipping"; continue; }
    REUSE=""
    [[ "$split" != "train" ]] && REUSE="--combined-weights-from $WNPZ"
    [[ -n "$REUSE" && ! -f "$WNPZ" ]] && { echo "  $ds: no weights from train, aborting"; exit 1; }
    echo "=== $ds/$split: $n members, weight set $WSET $(date +%H:%M)"
    "$PY" -m harmonize.ensemble_eval --mode featstack --test-h5 "$h5" \
        --pred-files $PF --rows first-day --dump-combined "$OUT" \
        --combined-weight-set "$WSET" $REUSE \
        || { echo "  ($ds/$split FAILED)"; exit 1; }
  done
  echo "done $ds $(date)"; exit 0
fi

# ---------------------------------------------------------------------------------------------
# WINDOWPROTO=1 — §38.1 experiment 2: the SAME model evaluated on all windows and on first-day
# windows only. Isolates the composition effect: an all-window evaluation lets a 20-day stay
# contribute ~31 windows and a 2-day stay ~3, so it is weighted toward long (and therefore sicker)
# patients. The model, the checkpoint and the split are identical; only which rows are scored
# changes, so the difference IS the bias.
#
#   WINDOWPROTO=1 sbatch --array=0-4 jobs/run_dump_preds.sh
if [[ "${WINDOWPROTO:-0}" == "1" ]]; then
  ALL=(umm hirid mimic_iii mimic_iv eicu)
  ds=${ALL[${SLURM_ARRAY_TASK_ID:-0}]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name="wproto_$ds" 2>/dev/null || true
  # The §16.9 val-selected denseAR winners, pinned by name. NOT taken from the pool manifest: its
  # denseAR slot is `ladderms_*` on MIMIC-III and a pteacher CONTROL on MIMIC-IV, neither of which
  # is the winner this experiment is about.
  case "$ds" in
    umm)       CK=hpsf_umm_6a82f5b0 ;;
    hirid)     CK=hpsf_hirid_f4e3a6c7 ;;
    mimic_iii) CK=hpsf_mimic_iii_095a5e97 ;;
    mimic_iv)  CK=hpsf_mimic_iv_88fb076a ;;
    eicu)      CK=hpsf_eicu_0fc6127a ;;
    *) echo "no winner pinned for $ds"; exit 1 ;;
  esac
  [[ -f "checkpoints/$CK/best.pt" ]] || { echo "missing checkpoint $CK"; exit 1; }
  h5="data/dataset/${ds}_seed42_test__all_4h_fwd.h5"
  for mode in all firstday; do
    FD=""; [[ "$mode" == "firstday" ]] && FD="--first-day-only"
    echo "=== $ds / $mode  (checkpoint $CK)  $(date +%H:%M)"
    # --out is MANDATORY here. Without it eval_forecast writes
    # docs/tables/eval_<run_name>.json, which is the PUBLISHED table for this winner; an untagged
    # re-run would overwrite it. That exact mistake cost a recovery-from-git once already.
    "$PY" -m harmonize.eval_forecast --checkpoint "checkpoints/$CK" --test-h5 "$h5" $FD \
        --out "docs/tables/eval_wproto_${ds}_${mode}.json" \
        || echo "  ($ds/$mode FAILED)"
  done
  echo "done $ds $(date)"; exit 0
fi

# ---------------------------------------------------------------------------------------------
# LGBMHEAD=1 — the 2x2 the §16.10 dissociation needs: {MLP, tree} x {input window, forecast}.
#
# The dissociation is currently measured against ONE discriminative architecture, _WindowEncoder,
# a flat MLP over the flattened window with no temporal structure. On UMM a gradient-boosted tree
# on the SAME input window already scores 0.8259 val / 0.7998 test, against the MLP's best FORECAST
# arm at 0.8214 / 0.7993 -- i.e. switching the classifier matches the entire "forecast gain". So
# the claim needs the second model class before it can be stated.
#
# Two modes per database, identical features, label and split:
#   window : features from the INPUT window            (tree, direct classification)
#   traj   : features from the LightGBM FORECAST       (tree, learned read-off)
# Together with the MLP's raw/frozen arms this completes the 2x2, so "does the forecast help?" can
# be asked within each model class rather than across them.
#
#   LGBMHEAD=1 sbatch --array=0-4 jobs/run_dump_preds.sh
if [[ "${LGBMHEAD:-0}" == "1" ]]; then
  ALL=(umm hirid mimic_iii mimic_iv eicu)
  ds=${ALL[${SLURM_ARRAY_TASK_ID:-0}]}
  [[ -n "${SLURM_JOB_ID:-}" ]] && scontrol update JobId="$SLURM_JOB_ID" Name="lgbmh_$ds" 2>/dev/null || true
  # TRAJSRC picks WHICH forecaster feeds the tree's read-off arm. This matters: the MLP arms use
  # the denseAR winner (hpsf_*), so a tree arm fed by the LightGBM forecaster is not a matched
  # comparison -- the forecaster differs as well as the classifier. TRAJSRC=densear closes that.
  case "${TRAJSRC:-lgbm}" in
    lgbm)    T="data/base_preds/lightgbmfix_traj_${ds}__${ds}" ;;
    densear) case "$ds" in
               umm)       T="data/base_preds/hpsf_umm_6a82f5b0__umm" ;;
               hirid)     T="data/base_preds/hpsf_hirid_f4e3a6c7__hirid" ;;
               mimic_iii) T="data/base_preds/hpsf_mimic_iii_095a5e97__mimic_iii" ;;
               mimic_iv)  T="data/base_preds/hpsf_mimic_iv_88fb076a__mimic_iv" ;;
               eicu)      T="data/base_preds/hpsf_eicu_0fc6127a__eicu" ;;
             esac ;;
    *) echo "unknown TRAJSRC ${TRAJSRC}"; exit 1 ;;
  esac
  MTAG="${TRAJSRC:-lgbm}"
  for head in ${HEADS:-mortality los3 los7}; do
    for seed in ${SEEDS:-42 123 456}; do
      for mode in window traj; do
        tagmode="$mode"; [[ "$mode" == "traj" && "$MTAG" != "lgbm" ]] && tagmode="traj${MTAG}"
        [[ "$mode" == "traj" && "${TRAJPM:-0}" == "1" ]] && tagmode="${tagmode}pm"
        dest="docs/tables/lgbmhead_${ds}_${head}_${tagmode}_s${seed}.json"
        [[ -f "$dest" ]] && { echo "  skip $head/$mode/s$seed (done)"; continue; }
        TJ=""
        [[ "$mode" == "traj" ]] && TJ="--traj-from ${T}_train_fd.npy,${T}_val_fd.npy,${T}_test_fd.npy"
        # TRAJPM=1 also appends the input window's count/since features, so the traj arm is not
        # handicapped by its dense mask (188 of 666 features are otherwise constant).
        [[ "$mode" == "traj" && "${TRAJPM:-0}" == "1" ]] && TJ="$TJ --traj-plus-mask"
        echo "--- $ds $head $mode s$seed $(date +%H:%M)"
        "$PY" -m harmonize.models.lightgbm_fc --dataset "$ds" --classify --head "$head" \
            --rows first-day --seed "$seed" --threads "${SLURM_CPUS_PER_TASK:-16}" $TJ \
            || echo "  ($ds $head $mode s$seed FAILED)"
      done
    done
  done
  echo "done $ds $(date)"; exit 0
fi
