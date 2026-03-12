# Boosting, Super Learner, and Competing-Risk Update

Date: 2026-03-12

## Scope

This update extends the AI-clinical-score training framework in three directions:

- Add `XGBoost-Surv`
- Add `LightGBM-Survival`
- Replace the previous random-search ensemble with a `super learner` style convex meta-learner
- Add a separate `Fine-Gray / competing-risk` branch

## Files changed

- `scripts/train_ai_clinical_score.py`
  - new base learners:
    - `xgboost_survival`
    - `lightgbm_survival`
  - both learners are calibrated through a one-dimensional Cox model so they expose `predict_survival_function`

- `scripts/train_manuscript_survival_quilts.py`
  - base learner list expanded from 9 to 11 models
  - new argument: `--ensemble-method`
  - default ensemble method is now `super_learner`
  - current supported methods:
    - `super_learner`
    - `random_search`

- `scripts/train_finegray_competing_risk.R`
  - new competing-risk branch
  - detects common RCC dataset column patterns automatically
  - currently validated on `AIdata/TCGA_CPTAC1_revised.csv`

## Current 11-model manuscript ensemble

The manuscript ensemble now uses:

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

## Super learner definition

The ensemble is now a horizon-specific convex meta-learner:

- weights are non-negative
- weights sum to 1
- one weight vector is learned for each horizon (`36/60/84/120` months)
- optimization uses constrained `SLSQP`
- objective: minimize horizon-specific IPCW Brier score on the validation set
- validation IPCW c-index is still reported for interpretation

## Smoke validation

### New boosting models

Smoke command:

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python - <<'PY'
from pathlib import Path
from scripts.train_ai_clinical_score import prepare_dataframe, fit_and_evaluate

df = prepare_dataframe(Path('AIdata/SEER.csv')).sample(n=1200, random_state=20260309).reset_index(drop=True)
metrics, summary = fit_and_evaluate(
    df,
    Path('/tmp/boost_surv_smoke'),
    Path('AIdata/SEER.csv'),
    model_names=['xgboost_survival', 'lightgbm_survival'],
)
print(metrics)
PY
```

Smoke result:

- `lightgbm_survival`: val IPCW c-index `0.5947`
- `xgboost_survival`: val IPCW c-index `0.5342`

### 11-model super learner

Smoke command:

```bash
/home/ubuntu/miniconda3/envs/starf/bin/python scripts/train_manuscript_survival_quilts.py \
  --input AIdata/SEER.csv \
  --output-dir /tmp/manuscript_superlearner_smoke \
  --sample-n 1200 \
  --random-draws 200 \
  --ensemble-method super_learner
```

Smoke result:

- validation `IPCW c-index @ 60 months`: `0.6604`
- test `IPCW c-index @ 60 months`: `0.6437`

### Fine-Gray branch

Smoke command:

```bash
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate r4.3
Rscript scripts/train_finegray_competing_risk.R \
  AIdata/TCGA_CPTAC1_revised.csv \
  /tmp/finegray_tcga_smoke2
```

Smoke result:

- dataset auto-detected:
  - overall endpoint: `time/status`
  - cause-specific endpoint: `CSS.time/CSS.status`
- inferred cause-time scale to overall time: `0.03285421`
- `n_event_interest`: `19`
- `n_competing_event`: `14`
- `n_censored`: `176`

## Important limitation

The current `AIdata/SEER.csv` file only contains a binary `status` column and does not include an explicit competing-event code. Therefore:

- `Fine-Gray` can be implemented as a branch in the framework
- but it cannot be trained on the current SEER training file without additional competing-risk labels
