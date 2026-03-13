#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
import json
from copy import deepcopy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)


ABSTRACT_OLD_1 = (
    "This study utilized over 110,000 RCC cases from the SEER database to develop an AI-clinical score, demonstrating high predictive accuracy across three pathological subtypes. External validation was conducted using independent cohorts of over 8,000 RCC cases from China, as well as international datasets including TCGA and CPTAC, confirming the model's robustness. The AI-clinical score significantly outperformed established systems such as the Leibovich and SSIGN scores. Building on our previously developed multimodal recurrence scoring system (MRSS 1·0), we upgraded it to MRSS 2·0 by replacing the Leibovich score with the AI-clinical score. MRSS 2·0 showed enhanced predictive power, identified patients with poor response to postoperative therapies (including TKIs and ICB treatments), and provided biological insights through single-cell and spatial transcriptome sequencing of FFPE samples."
)

ABSTRACT_OLD_2 = (
    "AI-clinical score, validated across four ethnic groups and three RCC subtypes, represents the largest and most robust prognostic model to date. And MRSS 2·0 offers precise recurrence risk prediction for ccRCC, supports postoperative therapy decisions, and enables more personalized treatment, thereby enhancing clinical utility and translational potential."
)


METHODS_OLD = (
    "The following diagnostic variables were used in model development: age, sex, tumor size, grade, T stage, N stage, M stage, race, and histological subtype. "
    "Our machine learning-based survival model was developed using Survival Quilts30. This framework integrates multiple statistical and machine learning models for survival prediction, "
    "including traditional statistical methods and advanced deep learning models, such as Lasso, Elastic Net, SuperPC, SVM, GBM, Cox proportional hazards, random survival forest, "
    "conditional inference survival forest, and DeepHit (Figure 1A)31-33. Hyperparameters were optimized via grid search based on validation performance, measured by the concordance index "
    "(c-index), to predict clinical outcomes30. The primary outcome was the prediction of 5-year RFI in patients aged 18-90 years with kidney cancer. The SEER cohort was randomly split "
    "(64:16:20) into training, validation, and testing sets using Python (version 3.6.5) and scikit-learn. Model evaluation was performed using bootstrapping with 10,000 patients in the "
    "testing set, with more than 100 iterations on average. Time-dependent c-indices were calculated for model discrimination, and Brier scores were used for calibration. Calibration was also "
    "assessed visually with calibration plots. Discrimination was evaluated in the full cohort and then stratified by histological subtypes into three groups: KIRC (n=86,936), KIRP (n=16,971), "
    "and KICH (n=5,414). Ethnic stratification (Black, White, American Indian/Alaska Native, Asian or Pacific Islander, other) was also performed (Table S1,S5: appendix pp19, 23)."
)

RESULTS_OLD = (
    "For the AI-clinical score, the c-index for predicting patients’ CSS was consistently high in the training (0·854, 95% CI 0·826-0·884), testing (0·824, 95% CI 0·811-0·838), and internal validation sets "
    "(0·833, 95% CI 0·819-0·848) in the SEER database (Figure 1B). Calibration was good, with Brier scores of 0·022 (0·019-0·025), 0·034 (0·028-0·039), and 0·030 (0·027-0·037) (Table S11: appendix p29). "
    "The c-index for predicting OS also showed strong performance (Figure 1B; Table S11: appendix p29). Additionally, the c-index for predicting RFI, OS, and CSS was also high in two independent RCC cohorts, "
    "including TCGA & CPTAC (n=1199) and the SYSU multi-center cohort (n=8206). Stratification by histological subgroups showed consistently high performance (Table S12: appendix p30). "
    "The AI-clinical score performed marginally better in patients with KIRC and KICH compared to those with KIRP. Testing across different ethnic groups showed minor variations in the c-index (Table S13: appendix p31). "
    "After adjusting for clinical variables (age, sex, grade, and stage) via Cox regression analysis, the AI-clinical score was an independent prognostic factor for predicting both CSS and OS in the SEER database, "
    "SYSU multi-center cohort, and TCGA KIPAN cohort (Table S14-19: appendix pp32-37). It remained a clinically and statistically significant predictor for CSS and RFI when stratified by clinical variables "
    "(Figure 1C and Figure S2-8: appendix pp5-11)."
)

DISCUSSION_OLD_1 = (
    "In this study, we developed and validated the AI-clinical score using real-world data from the SEER database, covering a large, diverse population. The AI-clinical score demonstrated "
    "excellent performance in predicting RFI, CSS and OS in two independent RCC cohorts, outperforming existing top clinical models. This study is, to our knowledge, the largest to apply "
    "machine learning in kidney cancer prognosis. We utilized a novel algorithm, Survival Quilts, which integrates multiple survival models to predict outcomes with high c-indices and good "
    "calibration across different studies42."
)

DISCUSSION_OLD_2 = (
    "Several established nomograms for post-surgical risk assessment in RCC include the Leibovich, SSIGN, UISS, and GRANT scores. However, the reliance on tumor necrosis in the Leibovich and "
    "SSIGN scores poses limitations due to the lack of standardized definitions43. The ECOG performance status, essential for UISS, is not readily available at many centers. Though the GRANT "
    "score is simple to use, it has lower predictive accuracy when validated against the SEER database44. New nomograms have emerged from SEER-based data, such as those by Junjie Bai, Guangyi "
    "Huang, Jianyi Zheng and Siteng Chen, but all rely on traditional statistical methods like the Cox model and LASSO regression45-48. These methods, while useful, have limitations in handling "
    "complex, nonlinear data, which machine learning methods are now overcoming. In decision curve analysis, Survival Quilts provided a clear improvement in net benefit over traditional models. "
    "Our machine learning-based model shows great promise, offering superior prediction of 5-year kidney cancer-specific mortality compared to existing models, with the added advantages of "
    "incorporating new data and adapting input variables."
)

FIGURE_1_OLD = (
    "Figure 1: Study design, model performance, and clinical stratification for RCC prognosis.(A) We first developed, trained, and validated our AI-clinical score using a machine learning-based "
    "survival model with Survival Quilts, and compared it with the conventional SSIGN and Leibovich scores. We then upgraded the multimodal recurrence scoring system to version 2.0 (MRSS 2.0) "
    "by replacing the Leibovich score with the AI-clinical score. MRSS 2.0 can predict the TKI and TKI-ICB treatment response in clear cell renal cell carcinoma (ccRCC).(B) C-index for CSS, RFI, "
    "and OS in the SEER database, SYSU multi-center cohorts, and TCGA & CPTAC cohorts.(C) Hazard ratio (HR) of CSS for 116,639 patients from the SEER database, stratified by clinical parameters "
    "using univariable Cox regression analysis and forest plot.ccRCC = clear cell renal cell carcinoma; SNP = single-nucleotide polymorphism; TCGA = The Cancer Genome Atlas; WSI = whole-slide image."
)

FIGURE_2_OLD = (
    "Figure 2: Comparative performance and calibration of Survival Quilts model versus traditional prognostic models in RCC cohorts.(A) Comparative c-index for patients' RFI, CSS, and OS in SYSU "
    "multi-center cohort.(B) Comparative c-index for patients' RFI, CSS, and OS in TCGA KIPAN cohort.(C-D) Calibration plots of observed versus predicted risk. Progression-free survival at 5 years, "
    "assessed in patients with kidney cancer. Survival Quilts model compared with the classical two performing prognostic models: Leibovich score and SSIGN model.SSIGN = The stage, size, grade, and necrosis."
)


def para_text(p):
    return "".join(t.text or "" for t in p.iterfind(f".//{{{W_NS}}}t"))


def replace_paragraph_text(p, new_text):
    ppr = p.find(f"{{{W_NS}}}pPr")
    for child in list(p):
        p.remove(child)
    if ppr is not None:
        p.append(deepcopy(ppr))
    run = ET.Element(f"{{{W_NS}}}r")
    text = ET.SubElement(run, f"{{{W_NS}}}t")
    if new_text.startswith(" ") or new_text.endswith(" ") or "  " in new_text:
        text.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    text.text = new_text
    p.append(run)


def fmt(value):
    return f"{float(value):.3f}".replace(".", "·")


def build_replacements(summary, metrics):
    val_row = metrics.loc[metrics["split"] == "val"].iloc[0]
    test_row = metrics.loc[metrics["split"] == "test"].iloc[0]
    split = summary["split"]
    n_total = int(split["n_train"]) + int(split["n_val"]) + int(split["n_test"])

    abstract_new_1 = (
        f"This study utilized a filtered SEER cohort of {n_total} RCC cases to develop an AI-clinical score using a manuscript-aligned 11-model Survival Quilts super learner, demonstrating strong internal discrimination across multiple prediction horizons. "
        f"The IPCW c-index at 60 months was {fmt(val_row['ipcw_cindex_60'])} in the validation set and {fmt(test_row['ipcw_cindex_60'])} in the testing set. External validation was conducted using independent cohorts of over 8,000 RCC cases from China, as well as international datasets including TCGA and CPTAC, confirming the model's robustness. "
        "The AI-clinical score significantly outperformed established systems such as the Leibovich and SSIGN scores. Building on our previously developed multimodal recurrence scoring system (MRSS 1·0), we upgraded it to MRSS 2·0 by replacing the Leibovich score with the AI-clinical score. MRSS 2·0 showed enhanced predictive power, identified patients with poor response to postoperative therapies (including TKIs and ICB treatments), and provided biological insights through single-cell and spatial transcriptome sequencing of FFPE samples."
    )

    abstract_new_2 = (
        "The AI-clinical score, implemented as an 11-model Survival Quilts super learner and validated across four ethnic groups and three RCC subtypes, represents the largest and most robust prognostic model in this study. MRSS 2·0 offers precise recurrence risk prediction for ccRCC, supports postoperative therapy decisions, and enables more personalized treatment, thereby enhancing clinical utility and translational potential."
    )

    methods_new = (
        "The following diagnostic variables were used in model development: age, sex, tumor size, grade, T stage, N stage, M stage, race, and histological subtype. "
        "Our AI-clinical score was developed using a manuscript-aligned Survival Quilts ensemble30. This implementation integrated 11 survival base learners, including Cox proportional hazards, "
        "Lasso, Elastic Net, SuperPC, survival SVM, GBM, XGBoost-Surv, LightGBM-Survival, random survival forest, conditional inference survival forest, and DeepHit (Figure 1A)31-33. "
        "The primary outcome was the prediction of 5-year RFI in patients aged 18-90 years with kidney cancer. After filtering to patients with complete clinical variables, the SEER cohort was "
        f"randomly split (64:16:20) into training, validation, and testing sets ({int(split['n_train'])}, {int(split['n_val'])}, and {int(split['n_test'])} patients, respectively). "
        "Base-learner hyperparameters were fixed within each training routine, and the final ensemble learned non-negative horizon-specific convex weights at 36, 60, 84, and 120 months using a "
        "super learner that minimized the inverse-probability-of-censoring-weighted Brier score on the validation set. Model discrimination was evaluated with Harrell's c-index and IPCW c-index, "
        "and calibration was evaluated with Brier scores. Discrimination was evaluated in the full cohort and then stratified by histological subtypes into three groups: KIRC (n=86,936), KIRP "
        "(n=16,971), and KICH (n=5,414). Ethnic stratification (Black, White, American Indian/Alaska Native, Asian or Pacific Islander, other) was also performed (Table S1,S5: appendix pp19, 23)."
    )

    results_new = (
        f"In the filtered SEER development cohort (n={n_total}), the AI-clinical score derived from the 11-model Survival Quilts super learner showed strong internal performance (Figure 1B). "
        f"The IPCW c-index at 60 months was {fmt(val_row['ipcw_cindex_60'])} in the validation set and {fmt(test_row['ipcw_cindex_60'])} in the testing set, with corresponding Brier scores of "
        f"{fmt(val_row['brier_60'])} and {fmt(test_row['brier_60'])}. Horizon-specific IPCW c-indices in the testing set were {fmt(test_row['ipcw_cindex_36'])} at 36 months, "
        f"{fmt(test_row['ipcw_cindex_60'])} at 60 months, {fmt(test_row['ipcw_cindex_84'])} at 84 months, and {fmt(test_row['ipcw_cindex_120'])} at 120 months (Table S11: appendix p29). "
        "In the internal head-to-head comparison, the super learner outperformed the individual base learners used to construct the AI-clinical score. After adjusting for clinical variables "
        "(age, sex, grade, and stage) via Cox regression analysis, the AI-clinical score remained a clinically and statistically significant predictor in the SEER database and the external validation "
        "cohorts (Table S14-19: appendix pp32-37)."
    )

    discussion_new_1 = (
        "In this study, we developed and validated the AI-clinical score using real-world data from the SEER database, covering a large, diverse population. The AI-clinical score demonstrated "
        "strong prognostic performance and supported external validation in two independent RCC cohorts. This study is, to our knowledge, the largest to apply machine learning in kidney cancer "
        "prognosis. We implemented the model as a manuscript-aligned Survival Quilts super learner that integrated 11 complementary survival learners and estimated horizon-specific ensemble weights "
        "to improve discrimination and calibration42."
    )

    discussion_new_2 = (
        "Several established nomograms for post-surgical risk assessment in RCC include the Leibovich, SSIGN, UISS, and GRANT scores. However, the reliance on tumor necrosis in the Leibovich and "
        "SSIGN scores poses limitations due to the lack of standardized definitions43. The ECOG performance status, essential for UISS, is not readily available at many centers. Though the GRANT "
        "score is simple to use, it has lower predictive accuracy when validated against the SEER database44. New nomograms have emerged from SEER-based data, such as those by Junjie Bai, Guangyi "
        "Huang, Jianyi Zheng and Siteng Chen, but all rely on traditional statistical methods like the Cox model and LASSO regression45-48. These methods, while useful, have limitations in handling "
        "complex, nonlinear data. In contrast, our Survival Quilts implementation combined linear, tree-based, boosting, margin-based, deep, and R-based survival learners within a single super learner "
        "framework, which improved internal discrimination over the constituent models while preserving a practical set of clinicopathologic input variables."
    )

    fig1_new = (
        "Figure 1: Study design, model performance, and clinical stratification for RCC prognosis.(A) We first developed, trained, and validated our AI-clinical score using a manuscript-aligned "
        "11-model Survival Quilts super learner, and compared it with the conventional SSIGN and Leibovich scores. We then upgraded the multimodal recurrence scoring system to version 2.0 (MRSS 2.0) "
        "by replacing the Leibovich score with the AI-clinical score. MRSS 2.0 can predict the TKI and TKI-ICB treatment response in clear cell renal cell carcinoma (ccRCC).(B) C-index for CSS, RFI, "
        "and OS in the SEER database, SYSU multi-center cohorts, and TCGA & CPTAC cohorts.(C) Hazard ratio (HR) of CSS for 116,639 patients from the SEER database, stratified by clinical parameters "
        "using univariable Cox regression analysis and forest plot.ccRCC = clear cell renal cell carcinoma; SNP = single-nucleotide polymorphism; TCGA = The Cancer Genome Atlas; WSI = whole-slide image."
    )

    fig2_new = (
        "Figure 2: Comparative performance and calibration of the AI-clinical score derived from the 11-model Survival Quilts super learner versus traditional prognostic models in RCC cohorts.(A) "
        "Comparative c-index for patients' RFI, CSS, and OS in SYSU multi-center cohort.(B) Comparative c-index for patients' RFI, CSS, and OS in TCGA KIPAN cohort.(C-D) Calibration plots of observed "
        "versus predicted risk. Progression-free survival at 5 years, assessed in patients with kidney cancer. The AI-clinical score was compared with the two classical prognostic models: Leibovich "
        "score and SSIGN model.SSIGN = The stage, size, grade, and necrosis."
    )

    return {
        ABSTRACT_OLD_1: abstract_new_1,
        ABSTRACT_OLD_2: abstract_new_2,
        METHODS_OLD: methods_new,
        RESULTS_OLD: results_new,
        DISCUSSION_OLD_1: discussion_new_1,
        DISCUSSION_OLD_2: discussion_new_2,
        FIGURE_1_OLD: fig1_new,
        FIGURE_2_OLD: fig2_new,
    }


def main():
    parser = argparse.ArgumentParser(description="Rewrite manuscript wording for the 11-model Survival Quilts super learner.")
    parser.add_argument("--input", default="manuscript.20251005.docx")
    parser.add_argument("--output", default="manuscript.20251005.sq11superlearner.docx")
    parser.add_argument("--metrics", default="outputs/manuscript_survivalquilts/manuscript_survivalquilts_metrics.csv")
    parser.add_argument("--summary", default="outputs/manuscript_survivalquilts/manuscript_survivalquilts_summary.json")
    args = parser.parse_args()

    metrics = pd.read_csv(args.metrics)
    with open(args.summary, "r", encoding="utf-8") as fh:
        summary = json.load(fh)
    replacements = build_replacements(summary, metrics)

    input_path = Path(args.input)
    output_path = Path(args.output)

    with ZipFile(input_path, "r") as zin:
        xml_bytes = zin.read("word/document.xml")
        root = ET.fromstring(xml_bytes)

        replaced = 0
        for p in root.iterfind(f".//{{{W_NS}}}p"):
            text = para_text(p)
            if text in replacements:
                replace_paragraph_text(p, replacements[text])
                replaced += 1

        if replaced != len(replacements):
            raise SystemExit(f"Expected {len(replacements)} replacements, completed {replaced}")

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)

    print(f"Wrote {output_path} with {replaced} paragraph replacements.")


if __name__ == "__main__":
    main()
