# Study Design Manifest

Date: 2026-03-09

Purpose: standardize cohort definitions, endpoint names, model names, and figure-to-data mapping for the MRSS 2.0 RCC manuscript package in this folder.

## 1. Canonical study structure

This project is best read as two linked studies plus one biology module:

1. AI-clinical score development for RCC prognosis
2. MRSS 2.0 construction for ccRCC recurrence and survival prediction
3. Postoperative therapy stratification and biology interpretation

Recommended canonical wording:

- AI-clinical score: machine-learning clinical prognostic model for RCC across histologic subtypes
- Comparator scores: Leibovich and SSIGN
- MRSS 2.0: multimodal recurrence score combining WSI-based score, six-SNP-based score, and AI-clinical score
- Disease scope:
  - RCC for AI-clinical score analyses
  - ccRCC for MRSS 2.0 and therapy analyses

## 2. Standardized cohort manifest

| Canonical cohort | Disease scope | Source file | Sheet | Observed size in file | Intended role | Main endpoints present | Notes |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| SEER RCC cohort | RCC, multi-histology | `SEER_revised.xlsx` | `SEER_revised` | 109,321 patients | AI-clinical score development and internal validation | CSS, OS | Headers include age, sex, race, subtype, grade, T/N/M, stage, tumor size, AI score columns |
| SYSU multi-center cohort | RCC, multi-histology | `SYSU_revised.xlsx` | `SYSU_revised` | 8,206 patients | External validation and head-to-head comparison | RFI/DFS, CSS, OS | Headers include necrosis, sarcomatoid, SSIGN, Leibovich, AI score columns |
| TCGA KIPAN + CPTAC cohort | RCC, multi-histology | `TCGA_CPTAC1_revised.xlsx` | `TCGA_CPTAC1_revised` | 1,199 patients | External validation and head-to-head comparison | RFI/RFS, CSS, OS | Headers include sarcomatoid, necrosis, AI score columns |
| MRSS combined cohort | ccRCC | `2750indep.xlsx` | `Sheet1` | 2,750 patients | Combined MRSS cohort summary | RFI/DFS, CSS, OS | Total matches manuscript-level MRSS population size |
| MRSS training set | ccRCC | `2750indep.xlsx` | `Sheet2` | 1,125 patients | MRSS 2.0 training | RFI/DFS, CSS, OS | Count matches manuscript statement for MRSS training set |
| MRSS independent validation set | ccRCC | `2750indep.xlsx` | `Sheet3` | 1,207 patients | MRSS 2.0 independent validation | RFI/DFS, CSS, OS | Count matches manuscript statement for independent validation set |
| TCGA-KIRC MRSS external set | ccRCC | likely derived from `2750indep.xlsx` combined sheet and/or separate upstream file | not frozen here | 418 patients in manuscript | MRSS 2.0 external validation | RFI, CSS, OS | 1,125 + 1,207 + 418 = 2,750, so the 418-case set is part of the combined MRSS total but is not isolated as a standalone sheet in this folder |
| Therapy cohort, pooled | RCC or high-risk ccRCC treatment series | `Treatment cohort.last.xlsx` | `Sheet1` | 127 patients | Therapy-response analyses and score export | PFS, CSS, OS | Includes SNPs, WSI score, AIscore60.final, `riskscore20241009`, `MRSS2.0`, treatment columns |
| Therapy cohort, TKI | postoperative Sunitinib series | `Treatment cohort.last.xlsx` | `TKI` | 66 patients | Figure 4A / treatment subgroup analyses | PFS, CSS, OS | Current workbook count conflicts with Table S4 draft count of 133 |
| Therapy cohort, TKI-ICB | postoperative immune-based combination series | `Treatment cohort.last.xlsx` | `TKI-ICB` | 61 patients | Figure 4B / treatment subgroup analyses | PFS, CSS, OS | Contains Axitinib-Toripalimab and Pembrolizumab-related cases |
| TG / treatment merge working file | ccRCC treatment benefit working subsets | `TG analysis cohort from 2750.xlsx` | multiple sheets | 67 to 666 rows depending on sheet | Derived treatment-benefit analyses | DFS, CSS, OS, MRSS2.0, treatment | Appears to be a working merge file rather than a frozen master dataset |
| TCGA revised Leibovich / SSIGN file | RCC, mainly benchmarking support | `TCGA936_revised.leibovich.SSIGN.xlsx` | `TCGA936_revised.leibovich.SSIGN` | 838 patients | Comparator-score support file | PFS, DSS/CSS-like, OS-related fields | Likely an older or parallel benchmarking dataset |

## 3. Standard endpoint dictionary

Use the following standard labels in all future writing:

- CSS: cancer-specific survival
- OS: overall survival
- RFI: recurrence-free interval
- RFS/DFS: use only if that exact term is locked to a source figure or table; otherwise normalize surgery cohorts to RFI
- PFS: progression-free survival for treatment cohorts

Observed naming drift that should be normalized:

- `RFI`, `RFS`, `DFS`, and even `PFS` are used interchangeably in several manuscript passages and filenames
- `SYSUPFS` and `2750PFS` file names appear to stand in for recurrence endpoints rather than metastatic progression
- Figure 4 legend contains merged labels such as `RFIDFS` and `DPFS`

Recommended rule:

- For SEER / SYSU / TCGA-CPTAC prognostic validation after surgery: use CSS and OS, and use RFI where recurrence is the endpoint
- For MRSS train / independent / TCGA-KIRC recurrence modeling: use RFI, CSS, and OS
- For postoperative treatment series: use PFS only when the cohort is genuinely progression-based; otherwise explicitly state DFS/RFI if the event is post-surgical recurrence

## 4. Standard model dictionary

- AI-clinical score
  - Scope: RCC prognosis
  - Inputs seen in source files: age, sex, race, histology, grade, T, N, M, stage, tumor size
  - Method named in manuscript: Survival Quilts

- Comparator scores
  - Leibovich score
  - SSIGN score

- MRSS 2.0
  - Scope: ccRCC recurrence and survival prediction
  - Components: WSI-based score, six-SNP-based score, AI-clinical score
  - Formula reported in manuscript:
    - `MRSS 2.0 = 3.5424 * WSI score + 1.1255 * six SNP-based score + 4.4860 * AI-clinical score - 4.0515`
  - Training cutoff reported in manuscript:
    - `risk score = 0`

## 5. Figure-to-data mapping

This section maps the current manuscript figure labels to the most likely source files available in this folder.

### Main figures

| Figure | Current file | Likely data source | Primary endpoint(s) | Notes |
| --- | --- | --- | --- | --- |
| Figure 1 | `Fig1.pdf` | `SEER_revised.xlsx`, `SYSU_revised.xlsx`, `TCGA_CPTAC1_revised.xlsx` | CSS, RFI, OS | Figure 1A is study schema; 1B is c-index summary; 1C is forest plot. Manuscript legend says 116,639 SEER CSS cases, which conflicts with current SEER file size and supplemental figure labels |
| Figure 2 | `Fig2.pdf` | `SYSU_revised.xlsx`, `TCGA_CPTAC1_revised.xlsx`, comparator-score inputs | RFI, CSS, OS | Comparative performance versus Leibovich and SSIGN |
| Figure 3 | `Fig3.pdf` | `2750indep.xlsx` and upstream WSI/SNP score inputs | RFI primarily, plus CSS and OS support in tables | Training set `n=1125`, independent set `n=1207`, TCGA-KIRC `n=418` are consistent with manuscript |
| Figure 4 | `Fig4.pdf` | `Treatment cohort.last.xlsx`, `TG analysis cohort from 2750.xlsx`, image assets | DFS/RFI/PFS, CSS, OS | Therapy stratification plus attention heatmaps and mIHC. Endpoint names are currently inconsistent in legend text |
| Figure 5 | `Fig5.pdf` | scRNA / spatial / response biology outputs, `RplotscRNA.pdf` support | meta-program biology | Single-cell and meta-program interpretation |

### Supplementary figures

| Supplementary figure | Current file | Likely data source | Intended endpoint | Notes |
| --- | --- | --- | --- | --- |
| Figure S1 | `Figure S1.pdf` | cohort assembly records | cohort flow | Patient selection process |
| Figure S2 | `Fig. S2.SEEROS.pdf` | `SEER_revised.xlsx` | OS | Matches appendix title |
| Figure S3 | `Fig. S3.SYSUCSS.pdf` | `SYSU_revised.xlsx` | OS or CSS | Conflict: appendix says SYSU OS; filename says SYSU CSS |
| Figure S4 | `FigS4.SYSUPFS.pdf` | `SYSU_revised.xlsx` | RFI | Conflict: appendix uses RFI; filename uses PFS |
| Figure S5 | `FigS5.SYSUOS.pdf` | `SYSU_revised.xlsx` | CSS | Conflict: appendix says CSS; filename says OS |
| Figure S6 | `FigS6.TCGACPTACOS.pdf` | `TCGA_CPTAC1_revised.xlsx` | OS | Consistent |
| Figure S7 | `FigS7.TCGACPTACRFS.pdf` | `TCGA_CPTAC1_revised.xlsx` | RFI/RFS | Terminology should be standardized |
| Figure S8 | `FigS8.TCGACPTACCSS.pdf` | `TCGA_CPTAC1_revised.xlsx` | CSS | Consistent |
| Figure S9 | `Fig S9.pdf` | `2750indep.xlsx` plus WSI/SNP/AI merged inputs | RFI model build / ROC / calibration / DCA | Appendix title aligns with MRSS 2.0 establishment and validation |
| Figure S10 | no exact final-number file found | `2750indep.xlsx` | OS | Current available file names still use older numbering |
| Figure S11 | no exact final-number file found | `2750indep.xlsx` | RFI | Current available file names still use older numbering |
| Figure S12 | no exact final-number file found | `2750indep.xlsx` | CSS | Current available file names still use older numbering |
| Figure S13 | `Fig S13.pdf` | `TG analysis cohort from 2750.xlsx`, `Treatment cohort.last.xlsx` | CSS and OS in treatment context | Appendix title: MRSS 2.0 high vs low after Sunitinib or Pembrolizumab initiation |
| Figure S14 | `Fig S14.pdf` | mIHC outputs | microenvironment biology | Distinct tumor microenvironment |
| Figure S15 | `Fig S15.pdf` | scRNA/meta-program outputs | angiogenesis / IFN-MHC programs | Biology support figure |

### Older-number supplementary files that likely need renaming or remapping

| Current file | Likely final intended figure number | Reason |
| --- | --- | --- |
| `FigS7.2750CSS.pdf` | probably Figure S12 | Appendix places 2750 CSS at S12 |
| `FigS8.2750PFS.pdf` | probably Figure S11 | Appendix places 2750 recurrence plot at S11 |
| `FigS9.2750OS.pdf` | probably Figure S10 | Appendix places 2750 OS at S10 |

## 6. Table-to-data mapping

| Table group | Likely source file(s) | Notes |
| --- | --- | --- |
| Table S1, S5-S7 | demographic and score-definition materials | demographic and algorithm support |
| Table S11-S19 | `SEER_revised.xlsx`, `SYSU_revised.xlsx`, `TCGA_CPTAC1_revised.xlsx` | AI-clinical score discrimination, calibration, and Cox analyses |
| Table S20-S25 | `2750indep.xlsx` and TCGA-KIRC external subset | MRSS 2.0 Cox analyses for RFI, CSS, and OS |
| Treatment tables | `Treatment cohort.last.xlsx`, `TG analysis cohort from 2750.xlsx`, center lists in supplementary tables docs | not fully frozen; numbering differs across document versions |

## 7. Conflicts that must be resolved before final revision

1. Figure numbering drift
   - The appendix assigns MRSS 2750 forest plots to Figures S10-S12.
   - Available files still use older labels `FigS7.2750CSS.pdf`, `FigS8.2750PFS.pdf`, and `FigS9.2750OS.pdf`.

2. Endpoint naming drift
   - `RFI`, `RFS`, `DFS`, and `PFS` are mixed across manuscript text, appendix titles, and PDF filenames.

3. SYSU supplementary figure label conflicts
   - The appendix titles and PDF filenames disagree for Figures S3-S5.

4. Treatment cohort definition is not frozen
   - `Treatment cohort.last.xlsx` currently contains 127 total rows, split into 66 TKI and 61 TKI-ICB rows.
   - `Table S4.docx` describes `Sunitinib (n=133)` and a `Propensity score matching cohort (with high risk of recurrence) (n=532)`.
   - `Supplementary Tables2-17.2025.docx` also contains treatment-center tables with a different numbering scheme.

5. Main-text legend drift
   - Figure 1C legend reports `116,639` SEER cases for CSS, while the current SEER workbook has 109,321 patient rows.
   - Figure 4 legend contains visibly merged or corrupted text fragments such as `RFIDFS`, `DPFS`, and duplicated regimen names.

## 8. Recommended frozen version list

If a clean package is needed for manuscript revision, use this as the provisional frozen source list:

- `manuscript.20251005.docx`
- `supplementary appendix.docx`
- `Supplementary Tables2-17.2025.docx`
- `SEER_revised.xlsx`
- `SYSU_revised.xlsx`
- `TCGA_CPTAC1_revised.xlsx`
- `2750indep.xlsx`
- `Treatment cohort.last.xlsx`
- `TG analysis cohort from 2750.xlsx`

## 9. Immediate next tasks

1. Freeze the final treatment cohort definition and decide whether the canonical counts are 127, 133, or a merged 532-case matched cohort.
2. Rename or remap the 2750 supplementary figure PDFs to the final S10-S12 numbering.
3. Standardize recurrence endpoint language across the manuscript to either `RFI` or a clearly justified alternative.
4. Reconcile the SYSU S3-S5 endpoint labels and the SEER Figure 1C sample count before further figure revision.
