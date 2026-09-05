# Prediction Model for Additional Procedure Requirement in Flexible Ureterorenoscopy Using Explainable Artificial Intelligence

This repository contains the complete analysis code accompanying the manuscript:

> **"Prediction Model for Additional Procedure Requirement in Flexible Ureterorenoscopy Using Explainable Artificial Intelligence"**  
> *Scientific Reports* (under revision)

---

## Overview

This study develops a machine learning-based clinical decision support system to predict the need for additional surgical procedures following flexible ureterorenoscopy (f-URS). The pipeline integrates nested cross-validation, explainability analyses, and a UPJ-PA-based risk stratification framework.

**Dataset:** 656 patients | 34 features | Binary outcome (additional procedure: Yes/No)  
**Class distribution:** 72.6% No (n=476) / 27.4% Yes (n=180)

---

## Repository Structure

```
├── ferhatHocaRIRSrevizyon1.ipynb   # Main analysis notebook
├── README.md
└── requirements.txt                # Python dependencies
```

---

## Analysis Pipeline

The notebook is organized into the following sequential stages:

| Stage | Description |
|-------|-------------|
| **Stage 1** | Data loading, preprocessing & exploratory data analysis (EDA) |
| **Stage 2** | Class balancing method comparison (ENN, SMOTE, etc.) |
| **Stage 3** | Feature selection comparison — Boruta & RFE (no data leakage) |
| **Stage 4** | Comprehensive model analysis with Nested Stratified Cross-Validation (outer: 5-fold, inner: 3-fold) |
| **Stage 5** | Model stability & bias analysis |
| **Stage 6** | Decision Curve Analysis (DCA) + ROC analysis |
| **Stage 7** | UPJ-PA robust model analysis |
| **Stage 8** | Explainability — SHAP, LIME, Permutation Importance, Model-Based Importance |
| **Stage 9** | Risk stratification by UPJ-PA deciles |
| **Stage 10** | Robustness analysis (noise injection & subgroup testing) |
| **Stage 11** | Clinical Decision Support System + Model export (`.pkl`) |

---

## Key Methodological Features

- **Nested Stratified K-Fold CV** (outer: 5-fold, inner: 3-fold) for unbiased performance estimation
- **EditedNearestNeighbours (ENN)** applied exclusively to training folds (no leakage)
- **100% original data** retained in test sets (no synthetic samples in evaluation)
- **95% Confidence Intervals** via t-distribution
- Overfitting detection (train-test gap, ratio, risk score)
- Calibration analysis (Brier Score, ECE, MCE, calibration curves)
- **XAI methods:** SHAP (TreeExplainer + LinearExplainer), LIME, Permutation Importance, LR Coefficients
- UPJ-PA-based clinical risk stratification

---

## Requirements

```bash
pip install -r requirements.txt
```

**Core libraries:**

```
pandas
numpy
scikit-learn
imbalanced-learn
xgboost
lightgbm
catboost
boruta
shap
lime
matplotlib
seaborn
scipy
openpyxl
```

---

## Usage

1. Clone the repository:
```bash
git clone https://github.com/<your-username>/fURS-additional-procedure-prediction.git
cd fURS-additional-procedure-prediction
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open the notebook:
```bash
jupyter notebook ferhatHocaRIRSrevizyon1.ipynb
```

> **Note:** The original patient dataset cannot be shared due to ethical and privacy restrictions in accordance with the Declaration of Helsinki. To reproduce the analysis, replace the data loading path in Stage 1 with your own dataset following the same variable naming convention.

---

## Citation

If you use this code in your research, please cite:

```
[Citation will be added upon publication]
```

---

## License

This code is released under the [MIT License](LICENSE).

---

## Contact

For questions regarding the code or methodology, please open an issue or contact the corresponding author.
