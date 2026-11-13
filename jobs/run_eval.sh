#!/bin/bash
#SBATCH --job-name=eval_forecast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/eval_%x_%j.out
#SBATCH --error=logs/eval_%x_%j.err
#
# GPU wrapper for the downstream clinical evaluation (Stage 5).
# Full test-set inference of the autoregressive decoder is slow on CPU, so run
# on a GPU node. All harmonize.eval_forecast CLI args are forwarded.
#
# Usage:
#   sbatch --partition=gpu-single --gres=gpu:1 --time=01:00:00 \
#       jobs/run_eval.sh \
#       --checkpoint checkpoints/<run> \
#       --test-h5 data/dataset/<ds>_seed42_test__all_4h_fwd.h5 \
#       --long-parquet data/long/<ds>_long.parquet

set -euo pipefail
WORKDIR="$(pwd)"
PYTHON="${PYTHON:-python}"
mkdir -p "${WORKDIR}/logs" "${WORKDIR}/docs/tables"

echo "Job started: $(date) on $(hostname)"
echo "Args: $*"
# ---------------------------------------------------------------------------
# RERUN=sepsis — refresh every stored eval whose Sepsis-3 block and SAPS-II
# coverage keys went stale when the label became time-dependent (§37.39).
# CPU-only: the clinical scores dominate and the decoder rollout is affordable
# at this size. Chunked so one array task does several evals in one allocation
# rather than spawning hundreds of tiny jobs.
#
#   sbatch --partition=cpu-single --time=24:00:00 --array=0-15 \
#       --export=ALL,RERUN=sepsis jobs/run_eval.sh
#
# NCHUNK must equal the array width. DRYRUN=1 prints the commands only.
# ---------------------------------------------------------------------------
if [[ "${RERUN:-}" == "sepsis" ]]; then
  NCHUNK="${NCHUNK:-16}"
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  # The work list is derived here, never stored, so it cannot go stale: every
  # docs/tables/eval_<run>.json that carries a sepsis3 block and still has a
  # checkpoint directory. FM/precomputed runs have no checkpoint and are skipped.
  LIST="$(${PYTHON} - <<'PYEOF'
import json, glob, os, itertools
by_ds = {}
for f in sorted(glob.glob("docs/tables/eval_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    if "sepsis3" not in d:
        continue
    run = os.path.basename(f)[5:-5]
    ds = d.get("dataset")
    if ds and os.path.isdir(os.path.join("checkpoints", run)):
        by_ds.setdefault(ds, []).append(run)
# Interleave the datasets before chunking. Cost per eval scales with window count
# (eICU 211k vs UMM 19.5k = 11x), so a dataset-sorted list would hand one chunk all
# the eICU runs and another all the UMM ones. Round-robin across datasets makes
# every chunk cost about the same.
for row in itertools.zip_longest(*(by_ds[k] for k in sorted(by_ds))):
    for ds, run in zip(sorted(by_ds), row):
        if run:
            print(f"{run}\t{ds}")
PYEOF
)"
  TOTAL=$(echo "$LIST" | grep -c . || true)
  echo "RERUN=sepsis: $TOTAL runs, chunk $IDX of $NCHUNK"
  n=0; done_n=0; fail_n=0
  while IFS=$'\t' read -r run ds; do
    [[ -z "$run" ]] && continue
    if (( n % NCHUNK == IDX )); then
      lp="data/long/${ds}_long.parquet"
      echo "[$(date +%H:%M:%S)] ($((done_n+1))) $run  ds=$ds"
      if [[ "${DRYRUN:-0}" == "1" ]]; then
        echo "    would run: eval_forecast --checkpoint checkpoints/$run --test-h5 data/dataset/${ds}_seed42_test__all_4h_fwd.h5 --long-parquet $lp"
      else
        # --out omitted on purpose: eval_forecast then writes the canonical
        # docs/tables/eval_<run_name>.json, i.e. it refreshes in place.
        ${PYTHON} -m harmonize.eval_forecast \
          --checkpoint "checkpoints/$run" \
          --test-h5 "data/dataset/${ds}_seed42_test__all_4h_fwd.h5" \
          ${lp:+--long-parquet "$lp"} || { echo "    FAILED: $run"; fail_n=$((fail_n+1)); }
      fi
      done_n=$((done_n+1))
    fi
    n=$((n+1))
  done <<< "$LIST"
  echo "chunk $IDX finished: $done_n evals, $fail_n failures"
  exit 0
fi

# ---------------------------------------------------------------------------
# RERUN=sepsis_heads — retrain the sepsis heads whose LABEL changed when the
# Sepsis-3 coincidence became time-dependent (§37.39). Each run is replayed from
# its own checkpoints/<run>/config.json, so the hyper-parameters are exactly the
# originals and only the label differs. train_head_cached runs are seconds (the FM
# embeddings are cached); the live frozen-backbone runs are the slow ones, so all of
# them go in ONE allocation rather than 23 tiny jobs.
#
#   sbatch --partition=cpu-single --time=24:00:00 \
#       --export=ALL,RERUN=sepsis_heads jobs/run_eval.sh
#
# DRYRUN=1 prints the reconstructed commands without running them.
# ---------------------------------------------------------------------------
if [[ "${RERUN:-}" == "sepsis_heads" ]]; then
  echo "RERUN=sepsis_heads: replaying every sepsis head from its stored config"
  ${PYTHON} - "${DRYRUN:-0}" <<'PYEOF'
import json, glob, os, subprocess, sys

dry = sys.argv[1] == "1"
PY = os.environ.get("PYTHON", sys.executable)
# checkpoints/ only: checkpoints_archive/ holds superseded copies that carry the SAME
# run_name, so including it would replay each name several times, every pass archiving
# the one before. De-duplicate on run_name for the same reason.
runs = sorted(d for d in glob.glob("checkpoints/*sepsis*") if os.path.isdir(d))
# ONLY=<run_name> replays exactly one run. The 2026-10-26 sequential job TIMED OUT after 24 h with
# 13 of 20 done, because a single live frozen-backbone head takes ~21.5 h on CPU
# (fb_sepsis_head_eicu_union). The remaining live heads must therefore each get their own
# allocation rather than sharing one.
_only = os.environ.get("ONLY", "").strip()
if _only:
    runs = [d for d in runs if os.path.basename(d) == _only]
    if not runs:
        print("ONLY=%r matched no run under checkpoints/*sepsis*" % _only); raise SystemExit(2)
# Cached-FM heads first: they finish in seconds off the stored embeddings, whereas the
# live frozen-backbone heads re-encode the backbone every epoch and may not finish on CPU.
# Cheap results then land early even if the tail has to be moved to a GPU.
runs.sort(key=lambda d: not os.path.basename(d).startswith("fmhead"))
ok = skipped = failed = 0
seen = set()
for d in runs:
    cfg_p = os.path.join(d, "config.json")
    if not os.path.exists(cfg_p):
        print("SKIP %s: no config.json (pre-dates config capture; superseded run)" % d)
        skipped += 1
        continue
    cfg = json.load(open(cfg_p))
    rn = cfg.get("run_name") or os.path.basename(d)
    if rn in seen:
        print("SKIP %s: run_name %r already replayed" % (d, rn))
        skipped += 1
        continue
    seen.add(rn)
    # train_head_cached is identified by its embedding caches; everything else is the
    # live train_head path. Both write vars(args), so every key maps to --key-with-dashes.
    mod = "harmonize.train_head_cached" if cfg.get("emb_train") else "harmonize.train_head"
    cmd = [PY, "-m", mod]
    for k, v in cfg.items():
        if v is None or v is False or v == "":
            continue
        flag = "--" + k.replace("_", "-")
        if v is True:
            cmd.append(flag)
        else:
            cmd += [flag, str(v)]
    print("\n=== %s  (%s)" % (d, mod.rsplit(".", 1)[1]))
    print("    " + " ".join(cmd[2:]))
    if dry:
        continue
    r = subprocess.run(cmd)
    if r.returncode == 0:
        ok += 1
    else:
        print("    FAILED rc=%d: %s" % (r.returncode, d))
        failed += 1
print("\nsepsis heads: %d rerun, %d failed, %d skipped (no config)" % (ok, failed, skipped))
PYEOF
  exit 0
fi

# ---------------------------------------------------------------------------
# RERUN=akifix — re-score every table with BOTH 2026-11-03 corrections:
#   (a) KDIGO's absolute >=0.3 mg/dL 48 h rise (D16), and
#   (b) the medication-zero CV fix (37.109), which is now the eval_forecast DEFAULT rather than
#       a flag -- an unobserved pressor was being LOCF-filled to its train mean, a therapeutic
#       dose, so the cardiovascular SOFA component read stage >=2 in 91 % of scored cells.
#       This one moves `sofa`, `matched_sofa*` and anything gated on dSOFA>=2, i.e. Sepsis-3.
#   A table is skipped only when it records aki_absolute_rise AND medication_zero.
#
# (original note on (a), retained)
# RERUN=akifix — re-score every table with KDIGO's absolute >=0.3 mg/dL 48 h rise (D16).
#   `eval_forecast` never passed creat_rise_48h, so the `aki` and `matched_aki*` blocks of all
#   526 tables describe severe RELATIVE azotemia (2.6-5.3 % prevalence) rather than KDIGO
#   stage 1 (9.5-15.3 %). For (a) alone nothing else in a table would change: kdigo_stage is
#   reached only from aki_comparison. **(b) is the opposite** and supersedes that sentence --
#   it moves every score that reads a pressor channel: sofa, matched_sofa*, dSOFA2*, daily_sofa*
#   and Sepsis-3. Only the `forecast` block stays byte-identical, because the error accumulators
#   run before the zeroing. Uses the same source resolution as RERUN=stale.
#
#       sbatch --array=0-15 --export=ALL,RERUN=akifix jobs/run_eval.sh
#
#   Tables already carrying aki_absolute_rise=true are skipped, so a timed-out array can be
#   resubmitted. Budget: the October full pass was 618 CPU-hours over 48 tasks.
#
# RERUN=stale — bring the tables that MISSED the 2026-10-26 recompute onto the
# corrected eval code (§37.76). 323 of 526 tables were regenerated with the
# reviewed SOFA/sepsis fixes (eb3fd1a observed-cell scoring, 3614f9a
# sofa_hour_mask, 006dffe CV/CNS masks, 449d1c8 time-dependent Sepsis-3); the
# other 203 were not, and the split falls along model families, so any table
# that puts a neural model beside an FM/classical baseline currently compares
# corrected scoring against defective scoring. This mode closes that gap.
#
# A stale table is one carrying NEITHER a sofa.*noCNS* key NOR
# forecast.hour_feat_mae — those two are the markers of the corrected eval.
#
# Each table is re-scored from whatever it was originally produced from:
#   checkpoints/<run_name>        -> --checkpoint   (133 tables)
#   data/fm_preds/<run_name>.npy  -> --pred-file    (25 tables, precomputed FM)
# Chronos/Chronos-2 (45 tables) had no full-split cache and were skipped and
# listed; submit_eval_chronos.sh MODE=refresh produces that cache in
# data/base_preds/<run_name>__<ds>_test.npy, which is the third source below, so
# a database becomes re-scorable as soon as its inference job has finished.
#
# --out is MANDATORY here and is the table's own filename. The identifier inside
# a table is run_name, which is NOT always the filename: eval_vol_gru_eicu.json
# carries run_name=gru_eicu_ms_s123, so letting eval_forecast default its output
# would overwrite eval_gru_eicu_ms_s123.json — a table that is ALREADY correct.
#
#   sbatch --partition=cpu-single --time=24:00:00 --array=0-15 \
#       --export=ALL,RERUN=stale jobs/run_eval.sh
#
# NCHUNK must equal the array width. DRYRUN=1 prints the commands only.
# ---------------------------------------------------------------------------
if [[ "${RERUN:-}" == "stale" || "${RERUN:-}" == "akifix" ]]; then
  # RERUN=akifix re-scores EVERY table with --aki-absolute-rise (D16). Identical
  # source resolution to RERUN=stale -- the only differences are that the staleness
  # filter is off and the flag is added. See the header block below.
  export RERUN_MODE="${RERUN}"
  NCHUNK="${NCHUNK:-16}"
  IDX="${SLURM_ARRAY_TASK_ID:-0}"
  # Derived here, never stored, so the work list cannot go stale.
  LIST="$(${PYTHON} - <<'INNER_PY'
import json, glob, os, itertools
import numpy as np, h5py

_NW = {}
def _windows(ds):
    """Window count of a dataset's test split, cached."""
    if ds not in _NW:
        try:
            with h5py.File(f"data/dataset/{ds}_seed42_test__all_4h_fwd.h5", "r") as f:
                _NW[ds] = f["x_in"].shape[0]
        except Exception:
            _NW[ds] = None
    return _NW[ds]

def covers(path, ds):
    """True only if a cached .npy exists AND has one row per window of this split.

    Existence is not enough. data/fm_preds holds leftover SMOKE/SUBSET dumps under the
    same naming convention -- every mira_*.npy is 6000 rows (mira_eicu_full.npy is a
    150000-row partial) against test splits of 19541-211251 windows. Routing a table to
    one of those made eval_forecast refuse it, correctly, as "predictions would be
    mis-paired with windows": 6 failures in the first pass of this mode.

    CORRECTION (2026-11-02): the earlier note here said the mira_* tables record the full
    window count "so the published numbers came from a full-split inference pass and only the
    cache is short". That was wrong, and the mistake was reading n_windows as coverage.
    run_mira.sh passes the same $lim_arg to BOTH stages, so the eval was limited too, while
    eval_forecast wrote n_windows = len(ds) regardless of --limit. The caches are not short --
    they are exactly as long as the runs were, and the TABLES overstated their coverage.
    eval_forecast now records n_windows = windows actually scored, plus n_windows_split and
    eval_limit. The five mira_<ds> tables are 6000-window smoke runs superseded by _full;
    mira_eicu_full covers 150000 of 211251 (71%) and the rest is being inferred as a shard.
    """
    if not os.path.exists(path):
        return False
    n = _windows(ds)
    try:
        rows = np.load(path, mmap_mode="r").shape[0]
    except Exception:
        return False
    return n is not None and rows == n
by_ds, skipped = {}, []
for f in sorted(glob.glob("docs/tables/eval_*.json")):
    try:
        d = json.load(open(f))
    except Exception:
        continue
    sofa = d.get("sofa") or {}
    fc = d.get("forecast") or {}
    # In akifix mode every table is a target, so the "already current" test is skipped; a table
    # that already carries aki_absolute_rise=true is the one to skip instead, so a killed array
    # can be resubmitted without redoing what landed.
    if os.environ.get("RERUN_MODE") == "akifix":
        # A table counts as done only when it carries BOTH corrections: the D16 AKI criterion
        # and the medication-zero CV fix (37.109). Keying on aki_absolute_rise alone would skip
        # a table re-scored before the CV fix landed and leave its SOFA block wrong.
        # Done tables stay IN the list (status "done") and are skipped by the loop, not here:
        # every array task rebuilds this list when it starts, so dropping them here shifted the
        # chunk boundaries between tasks that started hours apart -- the 2026-11-04 pass left 18
        # tables in no chunk. Numbering the full list keeps each table in one fixed chunk.
        done_tbl = d.get("aki_absolute_rise") is True and d.get("medication_zero") is True
    elif any("noCNS" in k for k in sofa) or "hour_feat_mae" in fc:
        continue                      # already on the corrected eval
    else:
        done_tbl = False
    tbl = os.path.basename(f)[5:-5]
    rn = d.get("run_name") or tbl
    ds = d.get("dataset") or ""
    # The table FILENAME may encode an evaluation VARIANT that the checkpoint
    # alone does not reproduce. Re-running such a table from the bare checkpoint
    # would silently replace a permuted/variant result with a plain one, so the
    # variant must either be reconstructed or the table skipped.
    extra = ""
    if tbl.startswith(rn) and tbl != rn:
        extra = tbl[len(rn):]
    # A variant that IS reconstructible: --permute-static <var> (the value may
    # itself contain underscores, e.g. weight_admission).
    perm = extra[len("_permstatic_"):] if extra.startswith("_permstatic_") else None
    # A trailing "_<dataset>" is benign: the FM tables are named
    # <run_name>_<ds> because one run is evaluated on each database's test H5.
    # It needs no extra flag, so do not treat it as an unreconstructible variant.
    if extra == "_" + ds:
        extra = ""
    # Named variants: suffixes whose meaning IS known, so the skip line says what the
    # table is instead of filing it as unexplained debt. "_capfix" is the CLIP-TO-VALID-CLASS
    # extraction of the eICU classical baselines (05_baselines.py line 325: "continuous
    # forecasts are rounded to the nearest valid class and clipped"; applied at line 448).
    # An unclipped ARIMA/ETS on a z-scored channel can diverge and a handful of blown-up
    # forecasts then dominate the MSE, so the clipped run is the CORRECT one -- and it is the
    # ONLY one: there is no eval_arima_eicu_4h.json to supersede it. report.py uses this
    # family for all four eICU numbers so they come from one consistent extraction. It is
    # therefore not re-scorable and not meant to be.
    NAMED = {"_capfix": "eICU classical baseline, clip-to-valid-class extraction "
                        "(05_baselines.py:325) -- the only version, not stale"}
    if extra in NAMED:
        skipped.append((tbl, NAMED[extra]))
        continue
    if extra and perm is None:
        # An unexplained suffix: a provenance difference not expressed by any eval flag
        # we can recover. Never guess it.
        skipped.append((tbl, (d.get("model") or "?") + " UNEXPLAINED variant" + extra))
        continue
    # Prefer an exact match on the table name over run_name: capfix-style
    # prediction caches are stored under the TABLE name.
    if os.path.isdir(os.path.join("checkpoints", tbl)):
        src = "ckpt\t" + os.path.join("checkpoints", tbl)
    elif covers(os.path.join("data/fm_preds", tbl + ".npy"), ds):
        src = "pred\t" + os.path.join("data/fm_preds", tbl + ".npy")
    elif os.path.isdir(os.path.join("checkpoints", rn)):
        src = "ckpt\t" + os.path.join("checkpoints", rn)
    elif covers(os.path.join("data/fm_preds", rn + ".npy"), ds):
        src = "pred\t" + os.path.join("data/fm_preds", rn + ".npy")
    elif covers(os.path.join("data/base_preds", "%s__%s_test.npy" % (rn, ds)), ds):
        # eval_fm_suite --dump-preds writes <run_name>__<ds>_<split>.npy here and
        # then SKIPS scoring, so a finished Chronos inference pass leaves a cache
        # and no table. covers() is what keeps the older first-day dumps
        # (*_fd.npy, a different name) and any half-written file out.
        src = "pred\t" + os.path.join("data/base_preds", "%s__%s_test.npy" % (rn, ds))
    else:
        skipped.append((tbl, d.get("model")))
        continue
    # The recorded test_h5 keeps variant inputs (e.g. the GTSQ-static *_gtsq.h5) -- the default
    # path rebuilds those models with the wrong static width. "-" keeps the IFS=tab split aligned.
    by_ds.setdefault(ds, []).append(("done" if done_tbl else "todo", tbl, ds,
                                     src + "\t" + (d.get("test_h5") or "-") + "\t" + (perm or "")))
# Round-robin across datasets before chunking: cost scales with window count
# (eICU 211k vs UMM 19.5k = 11x), so a dataset-sorted list would load one chunk
# with every eICU run and leave another with only UMM.
for row in itertools.zip_longest(*(by_ds[k] for k in sorted(by_ds))):
    for cell in row:
        if cell:
            print("%s\t%s\t%s\t%s" % cell)
import sys
for tbl, m in skipped:
    print("SKIP\t%s\t%s" % (tbl, m), file=sys.stderr)
print("skipped %d (no checkpoint, no cached preds)" % len(skipped), file=sys.stderr)
INNER_PY
)"
  TOTAL=$(echo "$LIST" | grep -c . || true)
  TODO=$(echo "$LIST" | grep -c "^todo" || true)
  echo "RERUN=${RERUN}: $TODO of $TOTAL re-scorable tables still to do, chunk $IDX of $NCHUNK"
  n=0; done_n=0; fail_n=0
  while IFS=$'\t' read -r status tbl ds kind path th5 perm; do
    [[ -z "$tbl" ]] && continue
    if (( n % NCHUNK == IDX )) && [[ "$status" == "todo" ]]; then
      lp="data/long/${ds}_long.parquet"
      h5="data/dataset/${ds}_seed42_test__all_4h_fwd.h5"
      [[ "$th5" != "-" && -f "$th5" ]] && h5="$th5"
      if [[ "$kind" == "ckpt" ]]; then SRC=(--checkpoint "$path"); else SRC=(--pred-file "$path"); fi
      [[ "${RERUN:-}" == "akifix" ]] && SRC+=(--aki-absolute-rise)
      # Reconstruct the permuted-static variant when the filename encodes one.
      [[ -n "${perm:-}" ]] && SRC+=(--permute-static "$perm")
      echo "[$(date +%H:%M:%S)] ($((done_n+1))) $tbl  ds=$ds  $kind=$path"
      if [[ "${DRYRUN:-0}" == "1" ]]; then
        echo "    would run: eval_forecast ${SRC[*]} --test-h5 $h5 --out docs/tables/eval_${tbl}.json"
      else
        ${PYTHON} -m harmonize.eval_forecast \
          "${SRC[@]}" \
          --test-h5 "$h5" \
          ${lp:+--long-parquet "$lp"} \
          --out "docs/tables/eval_${tbl}.json" \
          || { echo "    FAILED: $tbl"; fail_n=$((fail_n+1)); }
      fi
      done_n=$((done_n+1))
    fi
    n=$((n+1))
  done <<< "$LIST"
  echo "chunk $IDX finished: $done_n evals, $fail_n failures"
  exit 0
fi

${PYTHON} -m harmonize.eval_forecast "$@"
echo "Job finished: $(date)"
