# AI-Clinical Score Code Logic

Date: 2026-03-09

Purpose: reconstruct the intended code flow for AI-clinical score training from the manuscript text and the current workbook structure in this folder.

## 1. What the training code is supposed to do

The AI-clinical score is the RCC prognostic model trained on the SEER cohort using the Survival Quilts framework.

Per manuscript logic, the code should:

1. Load and clean the SEER RCC cohort
2. Build time-to-event prediction models using nine clinical variables
3. Tune Survival Quilts on a validation split
4. Generate 5-year risk predictions and time-dependent risk estimates
5. Benchmark against Leibovich and SSIGN
6. Export patient-level AI-clinical score values
7. Validate the score in SYSU and TCGA/CPTAC cohorts

## 2. Inputs defined by the manuscript

### Training cohort

- File: `SEER_revised.xlsx`
- Sheet: `SEER_revised`
- Current observed size: 109,321 patients

### Core predictors

The manuscript explicitly lists these variables:

1. age
2. sex
3. tumor size
4. grade
5. T stage
6. N stage
7. M stage
8. race
9. histological subtype

### Matching workbook columns in `SEER_revised.xlsx`

- `Age`
- `Sex`
- `Tumor Size Summary (2016+)`
- `Grade Pathological (2018+)`
- `Tsgemerge1`
- `Nstagemerge1`
- `Derived EOD 2018 M (2018+)`
- `Race recode (W, B, AI, API)`
- `Histologic Type ICD-O-3(1chRCC,2pRCC,3ccRCC)`

### Outcome columns currently present

- `time`
- `status`
- `CSSstatus`

Interpreted mapping:

- `time` + `status`: OS
- `time` + `CSSstatus`: CSS
- No clear recurrence endpoint exists in SEER, so the manuscript's use of `RFI` for SEER is likely wording drift rather than a literal train target

## 3. Most likely intended modeling target

Based on the manuscript and workbook structure, the cleanest interpretation is:

- Primary training targets in SEER:
  - CSS
  - OS

The workbook already contains derived columns:

- `AIscore0`
- `AIscore36`
- `AIscore60`
- `AIscore84`
- `AIscore120`
- `AIscore60.1`
- `AIscore60.2`

This strongly suggests the original code generated time-specific predicted risks or scores at:

- 0 months
- 36 months
- 60 months
- 84 months
- 120 months

For downstream RCC work, `AIscore60` appears to be the main exported feature used later in MRSS 2.0.

## 4. Training flow implied by the manuscript

### Step 1. Read and clean SEER data

Code module:

- `load_seer()`

Tasks:

1. Read `SEER_revised.xlsx`
2. Keep only eligible RCC patients
3. Remove patients with:
   - radiation
   - chemotherapy
   - missing survival data
   - missing tumor size
   - missing grade
   - missing stage
4. Keep complete data for:
   - subtype
   - size
   - grade
   - stage
   - OS
   - CSS
5. Restrict age to 18-90 years

Important note:

- The current workbook already looks post-filtered, so the original upstream exclusion code may have happened before `SEER_revised.xlsx` was written.

### Step 2. Encode predictors

Code module:

- `build_feature_matrix(df)`

Tasks:

1. Map categorical variables to model-ready form
2. Preserve subtype as a 3-level RCC histology factor:
   - 1 = chRCC
   - 2 = pRCC
   - 3 = ccRCC
3. Standardize or numerically encode continuous features:
   - age
   - tumor size
4. Encode ordinal clinical variables:
   - grade
   - T
   - N
   - M

Implementation choice:

- For reproducibility, one-hot encoding is safer than ad hoc integer coding for non-ordinal categories such as sex, race, and subtype.

### Step 3. Define survival targets

Code module:

- `build_survival_targets(df, endpoint)`

Expected targets:

- `endpoint = "css"`:
  - duration = `time`
  - event = `CSSstatus`

- `endpoint = "os"`:
  - duration = `time`
  - event = `status`

Important ambiguity:

- The manuscript sometimes writes that the AI-clinical score predicts `RFI`, `CSS`, and `OS`.
- The SEER workbook does not support true recurrence modeling.
- So code should probably train separate CSS and OS models in SEER, then reuse exported score structure across external datasets where recurrence exists.

## 5. Survival Quilts training logic

### Core model concept

The manuscript says Survival Quilts integrates multiple survival learners, including:

- Cox proportional hazards
- Lasso
- Elastic Net
- SuperPC
- SVM
- GBM
- random survival forest
- conditional inference survival forest
- DeepHit

So the code logic should be:

1. Fit a panel of candidate base learners on the training split
2. Use the validation split to tune hyperparameters
3. Combine or select the best risk predictions under the Survival Quilts framework
4. Export calibrated survival risk at 3, 5, 7, and 10 years, especially 5 years

### Data split

The manuscript explicitly states:

- SEER split = `64:16:20`
  - training
  - validation
  - testing

So code should use:

- `train_test_split` twice, with a fixed `random_state`

Recommended implementation:

1. Split full SEER into train 64% and temp 36%
2. Split temp into validation 16% and test 20%

### Hyperparameter tuning

The manuscript explicitly states:

- grid search
- validation performance measured by c-index

So code should have:

- `tune_survival_quilts(X_train, y_train, X_val, y_val)`

### Bootstrap evaluation

The manuscript states:

- bootstrapping with 10,000 patients in the testing set
- more than 100 iterations on average

So evaluation code should:

1. Sample 10,000 test patients with replacement
2. Compute c-index and Brier score
3. Repeat at least 100 times
4. Report mean and confidence intervals

## 6. Expected outputs of the training code

### Patient-level output table

Expected export from SEER training:

- patient ID
- survival time
- event indicators
- predicted survival/risk values at fixed horizons
- main AI-clinical score column for downstream work

Likely output columns:

- `AIscore36`
- `AIscore60`
- `AIscore84`
- `AIscore120`

### Model-level outputs

The manuscript implies these artifacts:

1. c-index summary
2. Brier score summary
3. calibration plots
4. decision curve analysis
5. subgroup performance tables by:
   - histology
   - ethnicity

### External validation outputs

The same trained scoring function should be applied to:

- `SYSU_revised.xlsx`
- `TCGA_CPTAC1_revised.xlsx`

Then evaluate:

- CSS
- OS
- recurrence endpoint where available

## 7. Comparator-code branch

In parallel to AI-clinical score training, the pipeline needs two classic-score branches:

### Leibovich branch

Inputs likely needed:

- T
- N
- tumor size
- grade
- necrosis

### SSIGN branch

Inputs likely needed:

- stage/size/grade/necrosis style variables

Code purpose:

1. Calculate fixed clinical scores
2. Predict 5-year endpoint risks
3. Compare discrimination and calibration against AI-clinical score

Important constraint:

- In SEER, necrosis is not obviously present in the current workbook.
- This means either:
  - comparator calculations were done on separate derived files, or
  - only subsets with available mapping were used, or
  - the current workbook is not the full modeling input used for those comparisons

## 8. Clean code architecture to implement

If we rebuild the training code now, the clean structure should be:

### `src/data_io.py`

- read xlsx
- harmonize column names
- define cohort schemas

### `src/preprocess.py`

- eligibility filtering
- missing-value filtering
- variable recoding
- train/val/test splitting

### `src/targets.py`

- build CSS target
- build OS target
- define fixed prediction horizons

### `src/sq_model.py`

- Survival Quilts model wrapper
- base learner registry
- hyperparameter grid
- fit / predict / save

### `src/evaluate.py`

- c-index
- time-dependent c-index
- Brier score
- bootstrap CIs
- calibration plots
- subgroup evaluation

### `src/comparators.py`

- Leibovich scoring
- SSIGN scoring

### `src/export_scores.py`

- export patient-level AI score tables
- write `AIscore60` and related horizon scores back to analysis tables

### `scripts/train_ai_clinical_score.py`

- entry point for SEER training

### `scripts/validate_ai_clinical_score.py`

- apply trained model to SYSU and TCGA/CPTAC

### `scripts/compare_with_clinical_scores.py`

- benchmark AI-clinical score vs Leibovich and SSIGN

## 9. Pseudocode for the actual training script

```python
seer = load_seer("SEER_revised.xlsx")
seer = filter_eligible_cases(seer)

X = build_feature_matrix(seer)
y_css = build_survival_targets(seer, endpoint="css")
y_os = build_survival_targets(seer, endpoint="os")

X_train, X_val, X_test, y_css_train, y_css_val, y_css_test = split_64_16_20(X, y_css)
_, _, _, y_os_train, y_os_val, y_os_test = split_same_indices(X, y_os)

sq_css = tune_and_fit_survival_quilts(X_train, y_css_train, X_val, y_css_val)
sq_os = tune_and_fit_survival_quilts(X_train, y_os_train, X_val, y_os_val)

seer["AIscore36_css"], seer["AIscore60_css"], seer["AIscore84_css"], seer["AIscore120_css"] = \
    predict_fixed_horizon_risks(sq_css, X, horizons=[36, 60, 84, 120])

seer["AIscore36_os"], seer["AIscore60_os"], seer["AIscore84_os"], seer["AIscore120_os"] = \
    predict_fixed_horizon_risks(sq_os, X, horizons=[36, 60, 84, 120])

css_metrics = bootstrap_evaluate(sq_css, X_test, y_css_test, n_boot=100, sample_size=10000)
os_metrics = bootstrap_evaluate(sq_os, X_test, y_os_test, n_boot=100, sample_size=10000)

save_metrics(css_metrics, "outputs/seer_css_metrics.csv")
save_metrics(os_metrics, "outputs/seer_os_metrics.csv")
save_scored_table(seer, "outputs/SEER_with_AI_clinical_scores.csv")
```

## 10. What is still ambiguous in the manuscript

These are the parts you should settle before writing real training code:

1. Is the AI-clinical score a single exported score, or separate endpoint-specific scores?
   - The workbook suggests horizon-specific scores
   - The manuscript often talks about one AI-clinical score used across endpoints

2. Was SEER used to train CSS only, OS only, or both?
   - Current columns support CSS and OS
   - True recurrence is not available in SEER

3. How were Leibovich and SSIGN computed in SEER?
   - Current SEER workbook does not clearly contain necrosis

4. What exact object was passed into MRSS 2.0?
   - Most likely `AIscore60`
   - This should be frozen explicitly

5. Were the `AIscore0/36/60/84/120` columns generated from one model or multiple endpoint-specific models?

## 11. Best practical conclusion for implementation

If we start coding now, the safest implementation path is:

1. Treat `AIscore60` as the canonical downstream AI-clinical score
2. Train separate Survival Quilts models for:
   - CSS
   - OS
3. Export fixed-horizon risks at 36, 60, 84, and 120 months
4. Use external datasets to test:
   - CSS
   - OS
   - recurrence endpoint where available
5. Defer strict Leibovich/SSIGN replication until the exact comparator input table is confirmed

This path is most consistent with both:

- the manuscript Methods
- the actual fields present in the current workbooks

