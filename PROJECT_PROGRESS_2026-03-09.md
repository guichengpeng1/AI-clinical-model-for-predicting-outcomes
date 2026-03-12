# Project Progress - 2026-03-09

## Current status

The project directory has been scanned at a high level.
The main study line is clear from the manuscript and appendix:

1. Develop an AI-clinical score for RCC prognosis.
2. Validate it across external cohorts and benchmark against Leibovich and SSIGN.
3. Upgrade MRSS 1.0 to MRSS 2.0 by replacing Leibovich with the AI-clinical score.
4. Test MRSS 2.0 for recurrence/survival prediction in ccRCC.
5. Evaluate whether MRSS 2.0 identifies patients who benefit from postoperative therapy.
6. Use pathology attention maps, mIHC, scRNA/spatial data for biological interpretation.

## Study framework

### Overall objective

Build and validate a clinically useful prognostic system for renal cell carcinoma, then extend it into a multimodal recurrence model for clear cell renal cell carcinoma that also supports postoperative treatment stratification.

### Main modules

1. AI-clinical score development
   - Development cohort: SEER
   - Candidate variables: age, sex, tumor size, grade, T stage, N stage, M stage, race, histologic subtype
   - Methods: Survival Quilts and comparison with standard survival models
   - Endpoints: RFI, CSS, OS
   - Metrics: c-index, Brier score, calibration, DCA

2. External validation and head-to-head comparison
   - Validation cohorts: SYSU multi-center, TCGA/CPTAC
   - Compare against Leibovich and SSIGN
   - Perform subgroup analysis by histology and ethnicity

3. MRSS 2.0 construction
   - Target disease: ccRCC
   - Modalities:
     - WSI-based score
     - six-SNP-based score
     - AI-clinical score
   - Goal: improve recurrence prediction beyond any single modality

4. MRSS 2.0 validation
   - Datasets: MRSS train/independent validation and TCGA-KIRC
   - Outputs: high/low risk groups, KM curves, AUC at 3/5/7 years, calibration, DCA

5. Therapy benefit stratification
   - Treatment cohorts include postoperative TKI and immune-based therapy related analyses
   - Goal: determine whether MRSS 2.0 identifies patients more likely to benefit

6. Biological interpretation
   - WSI attention heatmaps
   - multiplex IHC
   - single-cell transcriptomics
   - spatial transcriptomics
   - Focus: angiogenesis, antigen presentation, immune infiltration, tumor heterogeneity

## Key files already checked

- `manuscript.20251005.docx`
- `supplementary appendix.docx`
- `Supplementary Tables2-17.2025.docx`
- `SEER_revised.xlsx`
- `SYSU_revised.xlsx`
- `TCGA_CPTAC1_revised.xlsx`
- `2750indep.xlsx`
- `Treatment cohort.last.xlsx`
- `TG analysis cohort from 2750.xlsx`

## Important observations

1. This folder is mainly a results/manuscript packaging directory, not a complete reproducible analysis repository.
2. Only a few scripts are present, and they appear to be auxiliary or older exploratory scripts rather than the full modeling pipeline.
3. There are multiple versions of data files in the same folder.
4. Sample counts and treatment labels are not perfectly consistent across all files, especially in treatment-related datasets.
5. Before formal writing, figure revision, or downstream analysis, one master dataset version should be frozen.

## File-level data notes

### SEER_revised.xlsx

- Main sheet contains survival fields plus AI score related columns.
- Includes clinical variables and derived AI-clinical risk fields.
- Approximate size from sheet dimension: 109,322 rows including header.

### SYSU_revised.xlsx

- Contains DFS/CSS/OS related fields.
- Includes necrosis, sarcomatoid, SSIGN, Leibovich, and AI score related columns.
- Approximate size from sheet dimension: 8,207 rows including header.

### TCGA_CPTAC1_revised.xlsx

- Contains RFS/CSS/PFS/OS related fields and cohort source markers.
- Includes necrosis, sarcomatoid, AI score fields, and risk columns.
- Approximate size from sheet dimension: 1,200 rows including header.

### 2750indep.xlsx

- Appears to be the ccRCC MRSS cohort for multimodal validation.
- Contains OS, DFS, DSS, necrosis, stage, grade, riskScore, and risk.

### Treatment cohort.last.xlsx

- Includes treatment cohort sheets and multimodal variables:
  - SNP columns
  - WSI score
  - AIscore60.final
  - MRSS2.0
  - treatment status
- This file should be treated carefully because treatment definitions differ across project materials.

## Current interpretation of the paper logic

The paper can be organized as:

1. Clinical machine learning score development in large RCC cohorts
2. External validation and superiority over existing clinical scores
3. Multimodal upgrade to MRSS 2.0 in ccRCC
4. Clinical utility for recurrence prediction and adjuvant therapy selection
5. Mechanistic interpretation using tissue and transcriptomic assays

## Recommended next steps

1. Freeze a master version list for all cohorts and figures.
2. Confirm the exact treatment cohort definition used in the final manuscript.
3. Map each figure and supplementary table to its exact source dataset.
4. Write a one-page formal study design summary for manuscript/grant/PPT use.
5. If needed, reconstruct a clean analysis manifest linking:
   - input file
   - endpoint
   - cohort
   - analysis method
   - figure/table output

## Suggested immediate continuation point

If resuming later, the next efficient step is:

"Create a clean study design document that standardizes cohort definitions, endpoints, model names, and figure-to-data mapping."
