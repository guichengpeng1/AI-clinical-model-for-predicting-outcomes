# Manuscript Survival Quilts 9-Model Ensemble

Date: 2026-03-10

## Goal

Replace the recovered official `SurvivalQuilts` 6-model outer framework with a manuscript-aligned 9-model ensemble that matches the RCC paper methods list.

## Base models included

- `cox_ph`
- `lasso`
- `elastic_net`
- `survival_svm`
- `gbm`
- `random_survival_forest`
- `deephit`
- `superpc_r`
- `conditional_inference_survival_forest_r`

## Implementation

Training entry:

- `scripts/train_manuscript_survival_quilts.py`

Supporting changes:

- `scripts/train_ai_clinical_score.py`
  - added fixed `split manifest` support
  - added `seer_train_predictions.csv`
- `scripts/train_r_survival_models.R`
  - added fixed `split manifest` support
  - added `train` predictions for `SuperPC` and `cforest`

## Ensemble definition

- One common SEER split is generated first: `64:16:20`
- The 7 Python base models and 2 R base models are all trained on the same split
- For each prediction horizon (`36/60/84/120` months), the ensemble learns one non-negative weight vector over the 9 base models
- Weights are constrained to sum to 1
- Validation objective: maximize IPCW c-index at the corresponding horizon
- Search strategy: one-hot starts + uniform start + Dirichlet random search + focused second-stage Dirichlet refinement

## Important pragmatic detail

`survival_svm` does not output calibrated survival probabilities in the current local implementation. For horizon-specific integration, when a model lacks `risk_{horizon}` columns, the ensemble falls back to min-max normalized raw risk using the training-set scale. This keeps all 9 manuscript models inside the ensemble while preserving a horizon-specific weighting interface.

## Smoke run

Smoke command:

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python scripts/train_manuscript_survival_quilts.py \
  --input AIdata/SEER.csv \
  --output-dir /tmp/manuscript_survivalquilts_smoke \
  --sample-n 1000 \
  --random-draws 800
```

Smoke outputs:

- `/tmp/manuscript_survivalquilts_smoke/manuscript_survivalquilts_metrics.csv`
- `/tmp/manuscript_survivalquilts_smoke/manuscript_survivalquilts_weights.csv`
- `/tmp/manuscript_survivalquilts_smoke/manuscript_survivalquilts_bundle.joblib`

Smoke test metrics:

- `ipcw_cindex_36`: `0.687066`
- `ipcw_cindex_60`: `0.631228`
- `ipcw_cindex_84`: `0.574673`
- `ipcw_cindex_120`: `0.570976`

## Current status

- The manuscript-aligned 9-model ensemble pipeline now exists and runs end-to-end on local data
- A smoke run has completed successfully
- Full-cohort training has not yet been run in this note
