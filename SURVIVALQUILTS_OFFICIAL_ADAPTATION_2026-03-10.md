# Survival Quilts Official Adaptation

Date: 2026-03-10

## Source

- GitHub mirror: `https://github.com/vanderschaarlab/mlforhealthlabpub`
- Real code entry: `external/mlforhealthlabpub-main/alg/survivalquilts/`
- Key files:
  - `class_SurvivalQuilts.py`
  - `class_UnderlyingModels.py`
  - `utils_eval.py`

## What the official code actually contains

The official `SurvivalQuilts` implementation in `mlforhealthlabpub` is an earlier Python release of the temporal quilting framework. It does not wrap the 9 manuscript algorithms used in the RCC AI-clinical-score benchmark. Its built-in underlying models are:

- `CoxPH`
- `CoxPHRidge`
- `Weibull`
- `LogNormal`
- `LogLogistic`
- `RandomSurvForest`

`DeepHit`, `SuperPC`, `SVM`, `GBM`, and `conditional inference survival forest` are not part of this official `survivalquilts` module.

## Local adaptation

Adaptation script:

- `scripts/train_survival_quilts_official.py`

Local compatibility fixes applied in the adapter:

- Use the official `class_SurvivalQuilts.py` optimization logic as the outer framework.
- Replace the old `RandomSurvForest` prediction path with the current `sksurv` API using `unique_times_` and `predict_survival_function(..., return_array=True)`.
- Replace official internal non-stratified train/validation draws with stratified splits on event status to avoid fold-level all-censoring failures on SEER subsets.
- Reuse the local SEER preprocessing pipeline from `train_ai_clinical_score.py` so categorical predictors are one-hot encoded before entering the official framework.
- Keep the study-level split as `64:16:20` train/validation/test for local benchmarking.

## Smoke verification

Smoke command that completed successfully:

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python scripts/train_survival_quilts_official.py \
  --input AIdata/SEER.csv \
  --output-dir /tmp/survivalquilts_smoke \
  --sample-n 2000 \
  --K 3 \
  --num-bo 2 \
  --num-outer 1 \
  --num-cv 2 \
  --step-ahead 1
```

Smoke output summary:

- `n_total`: 2000
- `n_train / n_val / n_test`: `1280 / 320 / 400`
- test `ipcw_cindex_60`: `0.636259`
- test mean IPCW c-index across `36/60/84/120` months: `0.645240`

Smoke artifacts were written to:

- `/tmp/survivalquilts_smoke`

## Current status

- Official source archive has been recovered and extracted into the workspace.
- Official `SurvivalQuilts` outer framework now runs locally on the SEER-format data after compatibility adaptation.
- Full-cohort official training has not yet been completed in this note.
