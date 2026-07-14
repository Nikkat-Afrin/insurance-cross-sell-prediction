# Insurance Cross-Sell Prediction — KNN & SVM 🛡️

**Predict which existing insurance customers are most likely to buy an additional product, so the sales team can prioritize high-propensity cross-sell leads.**

[![CI](https://github.com/Nikkat-Afrin/insurance-cross-sell-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/Nikkat-Afrin/insurance-cross-sell-prediction/actions/workflows/ci.yml) ![Python](https://img.shields.io/badge/Python-3.12-blue) ![Models](https://img.shields.io/badge/Models-KNN%20%7C%20SVM%20%7C%20RandomForest-orange)
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 💼 Business problem
Cross-selling to existing customers is far cheaper than acquiring new ones. This project builds a classifier that scores each customer's likelihood of responding to a cross-sell offer (`TARGET` = Y/N), turning a 14k-customer book into a **ranked call list** for the sales team.

## 📊 Dataset
`data/M7_Data.csv` (bundled) — **14,016 customers × 15 features**, ~43% positive (cross-sell responders):

| Group | Features |
|---|---|
| Demographics | `age`, `age_P`, `city` |
| Relationship | `loyalty`, `LOR`, `lor_M` (length of relationship), `contract` |
| Product holdings | `prod_A`, `prod_B`, `type_A`, `type_B` |
| Value | `turnover_A`, `turnover_B` |
| Target | **`TARGET`** (Y = responds to cross-sell) |

## 🔬 Methodology
1. **EDA & preparation** — distribution/outlier checks, scaling, train/test split.
2. **Feature selection** — forward **Sequential Feature Selection** (mlxtend) scored on **F1**.
3. **Classification** — **K-Nearest Neighbors** and **Support Vector Machine** (the assignment focus), each tuned with `GridSearchCV`, plus Logistic Regression.
4. **Enhancement** — a reproducible comparison ([`src/model_comparison.py`](src/model_comparison.py)) that adds a **Random Forest** and reports accuracy, precision, recall, F1, and ROC-AUC side by side (SVM scored via decision function).

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Random Forest** | **0.933** | **0.917** | **0.928** | **0.922** | **0.974** |
| KNN (k=15) | 0.806 | 0.817 | 0.706 | 0.757 | 0.892 |
| SVM (RBF) | 0.807 | 0.791 | 0.749 | 0.769 | 0.883 |
| Logistic Regression | 0.751 | 0.705 | 0.725 | 0.715 | 0.829 |

<p align="center">
  <img src="reports/figures/roc_curves.png" width="48%" alt="ROC curves">
  <img src="reports/figures/feature_importance_rf.png" width="48%" alt="Feature importance">
</p>

**Takeaway:** KNN and SVM both deliver solid ranking power (ROC-AUC ≈ 0.88–0.89); adding a **Random Forest lifts ROC-AUC to 0.97** and F1 to 0.92, making it the model of choice for lead prioritization. Relationship length and product turnover are the strongest cross-sell signals.

## ▶️ How to run
```bash
pip install -r requirements.txt
jupyter lab notebooks/insurance_cross_sell_knn_svm.ipynb   # EDA + SFS + KNN/SVM (runs end-to-end)
python src/model_comparison.py                              # consolidated comparison + figures
```

## 🗂️ Structure
```
insurance-cross-sell-prediction/
├── data/M7_Data.csv
├── notebooks/insurance_cross_sell_knn_svm.ipynb
├── src/model_comparison.py
├── reports/{model_comparison.md, figures/}
├── requirements.txt
└── README.md
```

## 🛠️ Tech stack
`Python` · `pandas` · `scikit-learn` · `mlxtend` · `Matplotlib` · `Seaborn`

## 🚀 Future improvements
- Calibrated probabilities + a lift/decile table (what % of responders are captured in the top 2 deciles).
- Threshold tuned to the sales team's call capacity; gradient boosting (XGBoost/LightGBM) benchmark.

---
*Academic project (DAV 6150, Module 8), extended with a Random-Forest benchmark and a consolidated model comparison.*


## ⚖️ Fairness & calibration audit (`src/fairness_audit.py`)

High AUC isn't the same as deployable. The audit script asks the two questions a model reviewer would:

```bash
python src/fairness_audit.py     # -> reports/fairness_audit.md + figures
pytest tests/                    # audit logic tested on synthetic known-bias cases
```

- **Group fairness across age bands** — selection rate (demographic parity, screened with the 80% rule), TPR/FPR (equalized odds), precision, and AUC per group.
- **Finding worth knowing:** the ≤25 band is selected at only ~0.22× the rate of the 36–50 band. Its *base rate* is genuinely lower (14% vs 59%), so this is base-rate-driven rather than a pure model artifact — but it is exactly the kind of disparity an insurer must document and review before deployment. The audit makes it visible instead of leaving it buried in an aggregate AUC.
- **Calibration** — reliability table + Brier scores show whether "0.7 probability" really converts ~70% of the time; per-group Brier catches groups whose scores are less trustworthy.
- The audit logic is unit-tested against synthetic populations with *known* injected bias (flag must fire) and identical groups (flag must stay silent).
