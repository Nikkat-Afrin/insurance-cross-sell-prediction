"""Fairness and calibration audit for the cross-sell model.

A model with a great ROC-AUC can still be unusable: it may target one age
group far more aggressively than another (disparate treatment risk under
insurance regulation), or produce scores that don't mean what they say
(a "70% likely" lead that converts 40% of the time misleads the sales team).

This audit answers both questions for the champion Random Forest:

  * Group fairness across age bands — selection rate (demographic parity),
    TPR / FPR (equalized odds), precision, and AUC per group.
  * Calibration — reliability table + Brier score, overall and per group.

Outputs:
    reports/fairness_audit.md
    reports/figures/fairness_selection_rates.png
    reports/figures/calibration_curve.png

Run from the repo root:
    python src/fairness_audit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

RNG = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"

AGE_BANDS = [(0, 25, "<=25"), (26, 35, "26-35"), (36, 50, "36-50"),
             (51, 200, "51+")]

# Selection-rate ratio below this vs the most-selected group is flagged
# (the "80% rule" used in US employment-discrimination screening).
PARITY_FLOOR = 0.80


def age_band(age: pd.Series) -> pd.Series:
    out = pd.Series(index=age.index, dtype="object")
    for lo, hi, label in AGE_BANDS:
        out[(age >= lo) & (age <= hi)] = label
    return out


def load():
    df = pd.read_csv(ROOT / "data" / "M7_Data.csv").drop(columns=["ID"], errors="ignore")
    y = (df["TARGET"].astype(str).str.upper().str[0] == "Y").astype(int)
    X = df.drop(columns=["TARGET"])
    X = pd.get_dummies(X, columns=list(X.select_dtypes(include="object").columns),
                       drop_first=True, dtype=int)
    return X, y, df["age"]


def group_metrics(y_true: np.ndarray, proba: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    metrics = {
        "n": int(len(y_true)),
        "base_rate": round(float(np.mean(y_true)), 4),
        "selection_rate": round(float(np.mean(pred)), 4),
        "tpr": round(tp / (tp + fn), 4) if tp + fn else None,
        "fpr": round(fp / (fp + tn), 4) if fp + tn else None,
        "precision": round(tp / (tp + fp), 4) if tp + fp else None,
        "brier": round(float(brier_score_loss(y_true, proba)), 4),
    }
    if len(np.unique(y_true)) == 2:
        metrics["auc"] = round(float(roc_auc_score(y_true, proba)), 4)
    else:
        metrics["auc"] = None
    return metrics


def audit(y_true: pd.Series, proba: np.ndarray, groups: pd.Series,
          threshold: float = 0.5) -> dict:
    """Compute overall + per-group fairness metrics and parity flags."""
    pred = (proba >= threshold).astype(int)
    y_arr = y_true.to_numpy()
    report = {"threshold": threshold,
              "overall": group_metrics(y_arr, proba, pred),
              "groups": {}}
    for g in sorted(groups.dropna().unique()):
        idx = (groups == g).to_numpy()
        report["groups"][g] = group_metrics(y_arr[idx], proba[idx], pred[idx])

    rates = {g: m["selection_rate"] for g, m in report["groups"].items()}
    top = max(rates.values())
    report["parity"] = {
        "reference_group": max(rates, key=rates.get),
        "ratios": {g: round(r / top, 3) if top else None for g, r in rates.items()},
        "flagged": sorted(g for g, r in rates.items()
                          if top and r / top < PARITY_FLOOR),
        "rule": f"selection-rate ratio < {PARITY_FLOOR} vs most-selected group",
    }
    return report


def reliability_table(y_true: np.ndarray, proba: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    bins = np.clip((proba * n_bins).astype(int), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}",
                     "n": int(mask.sum()),
                     "mean_predicted": round(float(proba[mask].mean()), 4),
                     "observed_rate": round(float(y_true[mask].mean()), 4)})
    return pd.DataFrame(rows)


def write_report(report: dict, table: pd.DataFrame) -> Path:
    lines = ["# Fairness & Calibration Audit — Cross-Sell Model", "",
             f"Decision threshold: {report['threshold']}", "",
             "## Group metrics (age bands)", "",
             "| Group | n | Base rate | Selection rate | TPR | FPR | Precision | AUC | Brier |",
             "|---|---|---|---|---|---|---|---|---|"]
    def row(name, m):
        return (f"| {name} | {m['n']} | {m['base_rate']} | {m['selection_rate']} "
                f"| {m['tpr']} | {m['fpr']} | {m['precision']} | {m['auc']} | {m['brier']} |")
    lines.append(row("**Overall**", report["overall"]))
    for g, m in report["groups"].items():
        lines.append(row(g, m))
    parity = report["parity"]
    lines += ["", "## Demographic parity (80% rule)", "",
              f"Reference (most-selected) group: **{parity['reference_group']}**", "",
              "| Group | Selection-rate ratio |", "|---|---|"]
    for g, r in parity["ratios"].items():
        flag = " ⚠️" if g in parity["flagged"] else ""
        lines.append(f"| {g} | {r}{flag} |")
    lines += ["", f"Flagged groups: {parity['flagged'] or 'none'} ({parity['rule']})",
              "", "## Calibration (reliability table)", "",
              table.to_markdown(index=False), "",
              "_Interpretation: mean_predicted ≈ observed_rate per bin means the",
              "score is honest — a 0.7 score converts ~70% of the time._"]
    out = ROOT / "reports" / "fairness_audit.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> None:
    X, y, age = load()
    X_tr, X_te, y_tr, y_te, age_tr, age_te = train_test_split(
        X, y, age, test_size=0.25, stratify=y, random_state=RNG)

    model = RandomForestClassifier(n_estimators=300, random_state=RNG, n_jobs=-1)
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]

    report = audit(y_te, proba, age_band(age_te))
    table = reliability_table(y_te.to_numpy(), proba)
    out = write_report(report, table)

    FIG.mkdir(parents=True, exist_ok=True)
    groups = list(report["groups"])
    rates = [report["groups"][g]["selection_rate"] for g in groups]
    tprs = [report["groups"][g]["tpr"] for g in groups]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(x - 0.2, rates, 0.4, label="Selection rate")
    ax.bar(x + 0.2, tprs, 0.4, label="TPR (equal opportunity)")
    ax.set_xticks(x, groups); ax.set_xlabel("Age band"); ax.legend()
    ax.set_title("Selection rate & TPR by age band")
    fig.tight_layout(); fig.savefig(FIG / "fairness_selection_rates.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(table["mean_predicted"], table["observed_rate"], "o-")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed rate")
    ax.set_title("Calibration — Random Forest")
    fig.tight_layout(); fig.savefig(FIG / "calibration_curve.png", dpi=120)
    plt.close(fig)

    print(f"Audit written -> {out}")
    print(f"Parity flags: {report['parity']['flagged'] or 'none'}")


if __name__ == "__main__":
    main()
