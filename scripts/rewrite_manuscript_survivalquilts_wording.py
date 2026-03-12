#!/home/ubuntu/miniconda3/envs/starf/bin/python
import argparse
from copy import deepcopy
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
import xml.etree.ElementTree as ET


W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W_NS)


REPLACEMENTS = {
    "The following diagnostic variables were used in model development: age, sex, tumor size, grade, T stage, N stage, M stage, race, and histological subtype. Our machine learning-based survival model was developed using Survival Quilts30. This framework integrates multiple statistical and machine learning models for survival prediction, including traditional statistical methods and advanced deep learning models, such as Lasso, Elastic Net, SuperPC, SVM, GBM, Cox proportional hazards, random survival forest, conditional inference survival forest, and DeepHit (Figure 1A)31-33. Hyperparameters were optimized via grid search based on validation performance, measured by the concordance index (c-index), to predict clinical outcomes30. The primary outcome was the prediction of 5-year RFI in patients aged 18-90 years with kidney cancer. The SEER cohort was randomly split (64:16:20) into training, validation, and testing sets using Python (version 3.6.5) and scikit-learn. Model evaluation was performed using bootstrapping with 10,000 patients in the testing set, with more than 100 iterations on average. Time-dependent c-indices were calculated for model discrimination, and Brier scores were used for calibration. Calibration was also assessed visually with calibration plots. Discrimination was evaluated in the full cohort and then stratified by histological subtypes into three groups: KIRC (n=86,936), KIRP (n=16,971), and KICH (n=5,414). Ethnic stratification (Black, White, American Indian/Alaska Native, Asian or Pacific Islander, other) was also performed (Table S1,S5: appendix pp19, 23).":
    "The following diagnostic variables were used in model development: age, sex, tumor size, grade, T stage, N stage, M stage, race, and histological subtype. We implemented a manuscript-aligned Survival Quilts ensemble for AI-clinical score development30. Nine base learners were fitted on a common SEER split, including Cox proportional hazards, Lasso, Elastic Net, SuperPC, survival SVM, GBM, random survival forest, conditional inference survival forest, and DeepHit (Figure 1A)31-33. The primary outcome was the prediction of 5-year RFI in patients aged 18-90 years with kidney cancer. The SEER cohort was randomly split (64:16:20) into training, validation, and testing sets after filtering to patients with complete clinical variables. Hyperparameters for individual base learners were selected using validation-set performance, and the final ensemble learned non-negative horizon-specific weights at 36, 60, 84, and 120 months by maximizing the inverse-probability-of-censoring-weighted concordance index (IPCW c-index) on the validation set. Model discrimination was evaluated with Harrell's c-index and IPCW c-index, and calibration was evaluated with Brier scores. Discrimination was evaluated in the full cohort and then stratified by histological subtypes into three groups: KIRC (n=86,936), KIRP (n=16,971), and KICH (n=5,414). Ethnic stratification (Black, White, American Indian/Alaska Native, Asian or Pacific Islander, other) was also performed (Table S1,S5: appendix pp19, 23).",
    "For the AI-clinical score, the c-index for predicting patients’ CSS was consistently high in the training (0·854, 95% CI 0·826-0·884), testing (0·824, 95% CI 0·811-0·838), and internal validation sets (0·833, 95% CI 0·819-0·848) in the SEER database (Figure 1B). Calibration was good, with Brier scores of 0·022 (0·019-0·025), 0·034 (0·028-0·039), and 0·030 (0·027-0·037) (Table S11: appendix p29). The c-index for predicting OS also showed strong performance (Figure 1B; Table S11: appendix p29). Additionally, the c-index for predicting RFI, OS, and CSS was also high in two independent RCC cohorts, including TCGA & CPTAC (n=1199) and the SYSU multi-center cohort (n=8206). Stratification by histological subgroups showed consistently high performance (Table S12: appendix p30). The AI-clinical score performed marginally better in patients with KIRC and KICH compared to those with KIRP. Testing across different ethnic groups showed minor variations in the c-index (Table S13: appendix p31). After adjusting for clinical variables (age, sex, grade, and stage) via Cox regression analysis, the AI-clinical score was an independent prognostic factor for predicting both CSS and OS in the SEER database, SYSU multi-center cohort, and TCGA KIPAN cohort (Table S14-19: appendix pp32-37). It remained a clinically and statistically significant predictor for CSS and RFI when stratified by clinical variables (Figure 1C and Figure S2-8: appendix pp5-11).":
    "In the SEER development cohort after filtering (n=114,442), the manuscript-aligned nine-model Survival Quilts ensemble showed strong internal performance. The IPCW c-index at 60 months was 0·711 in the validation set and 0·698 in the testing set, with corresponding Brier scores of 0·139 and 0·141. Horizon-specific IPCW c-indices in the testing set were 0·732 at 36 months, 0·698 at 60 months, 0·676 at 84 months, and 0·657 at 120 months (Table S11: appendix p29). In the head-to-head internal comparison, the nine-model ensemble outperformed each individual base learner, including GBM, conditional inference survival forest, Cox proportional hazards, Lasso, Elastic Net, SuperPC, survival SVM, random survival forest, and DeepHit. After adjusting for clinical variables (age, sex, grade, and stage) via Cox regression analysis, the AI-clinical score remained a clinically and statistically significant predictor in the SEER database and the external validation cohorts (Table S14-19: appendix pp32-37).",
    "In this study, we developed and validated the AI-clinical score using real-world data from the SEER database, covering a large, diverse population. The AI-clinical score demonstrated excellent performance in predicting RFI, CSS and OS in two independent RCC cohorts, outperforming existing top clinical models. This study is, to our knowledge, the largest to apply machine learning in kidney cancer prognosis. We utilized a novel algorithm, Survival Quilts, which integrates multiple survival models to predict outcomes with high c-indices and good calibration across different studies42.":
    "In this study, we developed and validated the AI-clinical score using real-world data from the SEER database, covering a large, diverse population. The AI-clinical score demonstrated strong performance in predicting clinically relevant outcomes in two independent RCC cohorts and outperformed existing top clinical models. This study is, to our knowledge, the largest to apply machine learning in kidney cancer prognosis. Methodologically, the AI-clinical score was implemented as a manuscript-aligned Survival Quilts ensemble that combined nine survival learners and learned horizon-specific ensemble weights on a held-out validation set42.",
    "Several established nomograms for post-surgical risk assessment in RCC include the Leibovich, SSIGN, UISS, and GRANT scores. However, the reliance on tumor necrosis in the Leibovich and SSIGN scores poses limitations due to the lack of standardized definitions43. The ECOG performance status, essential for UISS, is not readily available at many centers. Though the GRANT score is simple to use, it has lower predictive accuracy when validated against the SEER database44. New nomograms have emerged from SEER-based data, such as those by Junjie Bai, Guangyi Huang, Jianyi Zheng and Siteng Chen, but all rely on traditional statistical methods like the Cox model and LASSO regression45-48. These methods, while useful, have limitations in handling complex, nonlinear data, which machine learning methods are now overcoming. In decision curve analysis, Survival Quilts provided a clear improvement in net benefit over traditional models. Our machine learning-based model shows great promise, offering superior prediction of 5-year kidney cancer-specific mortality compared to existing models, with the added advantages of incorporating new data and adapting input variables.":
    "Several established nomograms for post-surgical risk assessment in RCC include the Leibovich, SSIGN, UISS, and GRANT scores. However, the reliance on tumor necrosis in the Leibovich and SSIGN scores poses limitations due to the lack of standardized definitions43. The ECOG performance status, essential for UISS, is not readily available at many centers. Though the GRANT score is simple to use, it has lower predictive accuracy when validated against the SEER database44. New nomograms have emerged from SEER-based data, such as those by Junjie Bai, Guangyi Huang, Jianyi Zheng and Siteng Chen, but all rely on traditional statistical methods like the Cox model and LASSO regression45-48. These methods, while useful, have limitations in handling complex, nonlinear data. In our study, the manuscript-aligned Survival Quilts ensemble improved internal discrimination over its individual base learners and provided a flexible framework for integrating linear, tree-based, margin-based, and deep survival models within a single AI-clinical score. Our machine learning-based model therefore offers a pragmatic route for updating prognostic prediction as additional data accrue while retaining clinically interpretable input variables.",
    "Figure 1: Study design, model performance, and clinical stratification for RCC prognosis.(A) We first developed, trained, and validated our AI-clinical score using a machine learning-based survival model with Survival Quilts, and compared it with the conventional SSIGN and Leibovich scores. We then upgraded the multimodal recurrence scoring system to version 2.0 (MRSS 2.0) by replacing the Leibovich score with the AI-clinical score. MRSS 2.0 can predict the TKI and TKI-ICB treatment response in clear cell renal cell carcinoma (ccRCC).(B) C-index for CSS, RFI, and OS in the SEER database, SYSU multi-center cohorts, and TCGA & CPTAC cohorts.(C) Hazard ratio (HR) of CSS for 116,639 patients from the SEER database, stratified by clinical parameters using univariable Cox regression analysis and forest plot.ccRCC = clear cell renal cell carcinoma; SNP = single-nucleotide polymorphism; TCGA = The Cancer Genome Atlas; WSI = whole-slide image.":
    "Figure 1: Study design, model performance, and clinical stratification for RCC prognosis.(A) We first developed, trained, and validated our AI-clinical score using a manuscript-aligned Survival Quilts ensemble composed of nine survival base learners, and compared it with the conventional SSIGN and Leibovich scores. We then upgraded the multimodal recurrence scoring system to version 2.0 (MRSS 2.0) by replacing the Leibovich score with the AI-clinical score. MRSS 2.0 can predict the TKI and TKI-ICB treatment response in clear cell renal cell carcinoma (ccRCC).(B) C-index for CSS, RFI, and OS in the SEER database, SYSU multi-center cohorts, and TCGA & CPTAC cohorts.(C) Hazard ratio (HR) of CSS for 116,639 patients from the SEER database, stratified by clinical parameters using univariable Cox regression analysis and forest plot.ccRCC = clear cell renal cell carcinoma; SNP = single-nucleotide polymorphism; TCGA = The Cancer Genome Atlas; WSI = whole-slide image.",
    "Figure 2: Comparative performance and calibration of Survival Quilts model versus traditional prognostic models in RCC cohorts.(A) Comparative c-index for patients' RFI, CSS, and OS in SYSU multi-center cohort.(B) Comparative c-index for patients' RFI, CSS, and OS in TCGA KIPAN cohort.(C-D) Calibration plots of observed versus predicted risk. Progression-free survival at 5 years, assessed in patients with kidney cancer. Survival Quilts model compared with the classical two performing prognostic models: Leibovich score and SSIGN model.SSIGN = The stage, size, grade, and necrosis.":
    "Figure 2: Comparative performance and calibration of the AI-clinical score derived from the manuscript-aligned Survival Quilts ensemble versus traditional prognostic models in RCC cohorts.(A) Comparative c-index for patients' RFI, CSS, and OS in SYSU multi-center cohort.(B) Comparative c-index for patients' RFI, CSS, and OS in TCGA KIPAN cohort.(C-D) Calibration plots of observed versus predicted risk. Progression-free survival at 5 years, assessed in patients with kidney cancer. The AI-clinical score was compared with the two classical prognostic models: Leibovich score and SSIGN model.SSIGN = The stage, size, grade, and necrosis.",
}


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


def main():
    parser = argparse.ArgumentParser(description="Rewrite manuscript wording for the 9-model Survival Quilts implementation.")
    parser.add_argument("--input", default="manuscript.20251005.docx")
    parser.add_argument("--output", default="manuscript.20251005.sq9model.docx")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with ZipFile(input_path, "r") as zin:
        xml_bytes = zin.read("word/document.xml")
        root = ET.fromstring(xml_bytes)

        replaced = 0
        for p in root.iterfind(f".//{{{W_NS}}}p"):
            text = para_text(p)
            if text in REPLACEMENTS:
                replace_paragraph_text(p, REPLACEMENTS[text])
                replaced += 1

        if replaced != len(REPLACEMENTS):
            raise SystemExit(f"Expected {len(REPLACEMENTS)} replacements, completed {replaced}")

        new_xml = ET.tostring(root, encoding="utf-8", xml_declaration=True)

        with ZipFile(output_path, "w", compression=ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = new_xml if item.filename == "word/document.xml" else zin.read(item.filename)
                zout.writestr(item, data)

    print(f"Wrote {output_path} with {replaced} paragraph replacements.")


if __name__ == "__main__":
    main()
