# Fairness & Calibration Audit — Cross-Sell Model

Decision threshold: 0.5

## Group metrics (age bands)

| Group | n | Base rate | Selection rate | TPR | FPR | Precision | AUC | Brier |
|---|---|---|---|---|---|---|---|---|
| **Overall** | 3504 | 0.4292 | 0.4432 | 0.9468 | 0.0645 | 0.9169 | 0.9771 | 0.0553 |
| 26-35 | 1024 | 0.4922 | 0.5127 | 0.9524 | 0.0865 | 0.9143 | 0.9765 | 0.0603 |
| 36-50 | 1031 | 0.5946 | 0.614 | 0.9413 | 0.134 | 0.9115 | 0.9625 | 0.0776 |
| 51+ | 442 | 0.5498 | 0.5814 | 0.9671 | 0.1106 | 0.9144 | 0.9764 | 0.0647 |
| <=25 | 1007 | 0.143 | 0.137 | 0.9167 | 0.007 | 0.9565 | 0.9552 | 0.0232 |

## Demographic parity (80% rule)

Reference (most-selected) group: **36-50**

| Group | Selection-rate ratio |
|---|---|
| 26-35 | 0.835 |
| 36-50 | 1.0 |
| 51+ | 0.947 |
| <=25 | 0.223 ⚠️ |

Flagged groups: ['<=25'] (selection-rate ratio < 0.8 vs most-selected group)

## Calibration (reliability table)

| bin     |    n |   mean_predicted |   observed_rate |
|:--------|-----:|-----------------:|----------------:|
| 0.0-0.1 | 1210 |           0.0208 |          0.0198 |
| 0.1-0.2 |  308 |           0.1434 |          0.0779 |
| 0.2-0.3 |  194 |           0.2438 |          0.0515 |
| 0.3-0.4 |  134 |           0.3438 |          0.0896 |
| 0.4-0.5 |  105 |           0.45   |          0.0952 |
| 0.5-0.6 |   78 |           0.5469 |          0.2821 |
| 0.6-0.7 |  180 |           0.6549 |          0.7556 |
| 0.7-0.8 |  189 |           0.7427 |          0.8889 |
| 0.8-0.9 |  137 |           0.8443 |          0.9489 |
| 0.9-1.0 |  969 |           0.9886 |          0.999  |

_Interpretation: mean_predicted ≈ observed_rate per bin means the
score is honest — a 0.7 score converts ~70% of the time._