# AI Clinical Model for Predicting Outcomes

This repository tracks the RCC AI-clinical-score project, including training code, ensemble experiments, deployment assets, and manuscript-facing documentation.

## Repository Scope

Included here:

- `scripts/`
  - SEER training pipeline
  - manuscript-aligned Survival Quilts ensemble
  - R-side `SuperPC` and `conditional inference survival forest`
  - competing-risk `Fine-Gray` branch
  - manuscript/result synchronization utilities
- `webapp/`
  - lightweight AI-clinical-score calculator service
- `deploy/`
  - `systemd`, `nginx`, and startup templates
- selected result summaries under `outputs/`
- project notes and the latest manuscript draft

Excluded here:

- raw cohort data under `AIdata/`
- large figure binaries and local office assets
- exploratory local folders and oversized PDFs

## Current Ensemble Stack

The current manuscript ensemble supports:

- `cox_ph`
- `lasso`
- `elastic_net`
- `survival_svm`
- `gbm`
- `xgboost_survival`
- `lightgbm_survival`
- `random_survival_forest`
- `deephit`
- `superpc_r`
- `conditional_inference_survival_forest_r`

The ensemble currently supports:

- `super_learner`
- `random_search`

## Key Artifacts

- unified comparison:
  - `outputs/model_comparison/seer_model_metrics_unified.csv`
- manuscript ensemble metrics:
  - `outputs/manuscript_survivalquilts/manuscript_survivalquilts_metrics.csv`
- manuscript ensemble weights:
  - `outputs/manuscript_survivalquilts/manuscript_survivalquilts_weights.csv`
- updated manuscript:
  - `manuscript.20251005.sq9model.docx`

## Deployment

See:

- `webapp/README.md`
- `deploy/DEPLOY_WEBAPP.md`
