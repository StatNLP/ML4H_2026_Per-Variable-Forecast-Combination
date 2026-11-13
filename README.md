# Which Model, For Which Variable? — Per-Variable Forecast Combination in Sparse ICU Time Series

Code for the ML4H 2026 paper. Fourteen forecasters (seventeen in the extended pool) predict the next
24 hours of a 94-channel ICU panel from the previous 24 hours, on five databases: MIMIC-III,
MIMIC-IV, eICU, HiRID and the UMM cohort (Universitätsmedizin Mannheim). Their cached predictions
are combined with weights fitted separately for every output variable.

## What is here and what is not

**Here:** everything needed to go from the raw databases to every number, table and figure:
extraction and harmonisation, gridding, base-model training and hyperparameter sweeps, foundation-
model inference, prediction caching, the combiners, the uncertainty analyses and the appendix and
figure generators, together with the job scripts that ran them.

**Not here:**

- **Data.** MIMIC-III, MIMIC-IV, eICU and HiRID are available on PhysioNet after credentialing and
  signing the data use agreement. They cannot be redistributed. The UMM cohort cannot be shared.
- **Trained checkpoints and cached predictions**, because they are derived from these data.

The pipeline therefore runs end to end for anyone with access to the four public databases.

## Environment

Three Python environments are needed, because the two foundation models pin incompatible versions
of PyTorch and NumPy:

| environment | Python | requirements | used for | interpreter variable |
|---|---|---|---|---|
| main | 3.13 | `requirements.txt` | everything except the two foundation models | `PYTHON` |
| TiRex-2 | 3.13 | `requirements-tirex2.txt` | `harmonize/infer_tirex2.py` (`jobs/submit_eval_tirex2.sh`) | `PY_TIREX` |
| Moirai | 3.11 | `requirements-moirai.txt` | `harmonize/infer_moirai.py` (`jobs/run_infer_moirai.sh`) | `PYTHON_MOIRAI` |

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
python3.13 -m venv .venv-tirex2 && .venv-tirex2/bin/pip install -r requirements-tirex2.txt
python3.11 -m venv .venv-moirai && .venv-moirai/bin/pip install -r requirements-moirai.txt

export PROJECT_ROOT=$(pwd) PYTHONPATH=$(pwd)
export PYTHON=.venv/bin/python PY_TIREX=.venv-tirex2/bin/python PYTHON_MOIRAI=.venv-moirai/bin/python
```

The job scripts fall back to `python` on the `PATH` when these variables are not set.

The scripts in `jobs/` are SLURM wrappers. Each is a thin loop around one `python -m harmonize.…`
call, so they also run without SLURM (`bash jobs/<script>.sh`). Partition names in their
`#SBATCH` headers are those of our cluster.

## Layout

```
harmonize/
  mimic_iii.py mimic_iv.py eicu.py hirid.py umm.py   per-database extraction to one long table
  stage2_preprocessing.py                            hourly aggregation (mean per hour since ICU
                                                     admission), normalisation, patient-level split
  stage3_make_dataset.py                             24 h input / 24 h target windows at a 4 h stride (HDF5)
  stage3_make_triplets.py triplet_dataset.py         triplet view for STraTS
  icu_dataset.py common.py                           loaders and shared helpers
  models/                                            all base models (one file per family) + LightGBM
  hpsweep/                                           expands a sweep grid into one training job per configuration
  train_forecast.py eval_forecast.py                 training and scoring of the base models
  infer_moirai.py infer_tirex2.py                    zero-shot foundation-model inference
  featstack.py                                       the combiners (closed form from cached statistics)
  ensemble_eval.py                                   fits and scores the combiners; development and final protocols
  scores/ feature_selection.py diagnoses.py          clinical scores and helpers imported by the above
configs/
  features.yaml                                      the 94-channel catalogue (type, unit, organ system)
  sweeps/                                            hyperparameter grids, one line per configuration
  pool_manifest.tsv                                  the 12 trained pool members per database
  checkpoints_manifest.tsv                           configuration and seed of each selected member
  trained_configurations.tsv                         number of trained runs per family and database
10_scripts/                                          figures, appendix tables and supporting analyses
jobs/                                                job scripts (see the pipeline below)
```

`10_scripts/features.yaml` is a copy of `configs/features.yaml`, at the path the combiner reads it
from for the clinical group labels.

## Pipeline

| step | command | output |
|---|---|---|
| 1. extraction | `bash jobs/run_stage1.sh <dataset>` | `data/long/<dataset>_long.parquet` |
| 2. preprocessing | `bash jobs/run_stage2.sh` | hourly table, normalisation statistics, patient-level 64/16/20 split |
| 3. windows | `bash jobs/run_stage3.sh` (STraTS: `jobs/run_make_triplets.sh`) | `data/dataset/<dataset>_seed42_{train,val,test}__all_4h_fwd.h5` |
| 4. training | `python -m harmonize.hpsweep.submit configs/sweeps/<grid>.txt --wrapper jobs/run_sweep_one.sh`, single runs with `jobs/run_train_gpu.sh`, seeds with `jobs/run_seed_bundle.sh` | `checkpoints/<run>/` |
| 5. foundation models | `jobs/run_infer_moirai.sh`, `jobs/submit_eval_tirex2.sh` | cached zero-shot predictions |
| 6. cached predictions | `bash jobs/submit_dump_preds.sh` | `data/base_preds/<member>__<dataset>_{val,test}.npy` |
| 7. combiners, development | `MODE=dev bash jobs/run_final_test.sh` | fit on one half of the validation patients, score on the other |
| 8. combiners, final | `bash jobs/run_final_test.sh` | fit on all of validation, score test once |
| 9. appendix analyses | `bash jobs/run_final_extras.sh` | uncertainty, paired tests, win shares, per-system and per-horizon tables |
| 10. tables and figures | `python 10_scripts/make_appendix.py`, `python 10_scripts/analyze_error_structure.py` | `docs/tables/`, `docs/figures/` |

The pool is defined by `configs/pool_manifest.tsv`, never by a directory listing, so additional
checkpoints in `data/base_preds/` do not change it.

## Protocol

Combiners are compared under a development protocol: the combiner is fitted on one stay-grouped half
of the validation split and scored on the other half. The selected combiner (per-variable simplex
weights shrunk toward a harmonic rank prior, `simplex_shrunk500_var`) was then fitted on the whole
validation split and scored once on test. Analyses added afterwards either re-analyse the existing
test predictions (resampling, paired tests, threshold sweeps) or use the development protocol.

## Analyses added during the review

| analysis | command |
|---|---|
| patient-level bootstrap, paired tests (incl. harm-14), random-weight baseline | `bash jobs/run_final_extras.sh` |
| per-channel support and sparsity-threshold sweep | `SUPPORT=1 bash jobs/run_final_extras.sh` |
| pool without the foundation models | `NOFM=1 bash jobs/run_final_test.sh` (development), `NOFM=1 RBTAG=nofm12 bash jobs/run_final_extras.sh` (test) |
| seed pools (variance control) | `SEEDPOOL=strats bash jobs/run_final_test.sh`, `SEEDPOOL=dense_autoregressive …` |
| LightGBM, ARIMA, ETS members | `LGBM=1 LGBM_ARGS="--fixed-rounds 200" sbatch --array=0-4 jobs/run_dump_preds.sh`, `CLASSIC=1 sbatch --array=0-4 jobs/run_dump_preds.sh` |
| extended pools | `WIDE=1 ADD="arima ets lightgbmfix" bash jobs/run_final_test.sh`; test: `ADD="arima ets lightgbmfix" RBTAG=wide17 bash jobs/run_final_extras.sh` |
| random subsets of k members | `KABL="2 4 6 8 10 12" DRAWS=5 bash jobs/run_final_test.sh` |
| ridge penalty sweep | `POOL14=1 RBTAG=ridge_10 DEVARGS="--ridge-lambda 10" bash jobs/run_final_test.sh` |
| measurement intensity, per-model × group tables | `RBX=1 POOL14=1 bash jobs/run_final_test.sh` |

Runs with an `RBTAG` write to their own tables (`docs/tables/*_rb_<tag>*`, `docs/tables/final_rb_<tag>/`)
and never overwrite the published ones.

## Citation

```bibtex
@inproceedings{FracarolliRiezler:26:WhichModel,
  title     = {Which Model, For Which Variable? Per-Variable Forecast Combination in Sparse ICU Time Series},
  author    = {Fracarolli, Marius and Riezler, Stefan},
  booktitle = {Machine Learning for Health (ML4H) -- Findings Track},
  year      = {2026}
}
```
