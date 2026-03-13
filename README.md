# AI Clinical Model for Predicting Outcomes

This repository tracks the RCC AI-clinical-score codebase, including training pipelines, ensemble experiments, and deployment assets.

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

Excluded here:

- raw cohort data under `AIdata/`
- large figure binaries and local office assets
- generated outputs, manuscript drafts, and exploratory local folders

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

The ensemble meta-learner currently supports:

- `super_learner`
- `random_search`

## Local Runtime Outputs

Formal training outputs are written locally under `outputs/` and are intentionally git-ignored.

## Deployment

Primary entry points:

- `deploy/start_ai_clinical_score_web.sh`
- `deploy/systemd/ai-clinical-score.service`
- `deploy/nginx/ai-clinical-score.conf`
