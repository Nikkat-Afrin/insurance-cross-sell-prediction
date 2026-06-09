"""
Insurance Cross-Sell — reproducible model comparison (KNN, SVM, LogReg, RF).

Enhancement over the notebook: a single side-by-side comparison with ROC-AUC
(using decision scores for SVM), confusion matrix, and RF feature importance.
Writes reports/model_comparison.md and reports/figures/*.png.
Run from repo root:  python src/model_comparison.py
"""
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix, ConfusionMatrixDisplay)

warnings.filterwarnings("ignore")
RNG = 42
ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def scores(model, X):
    """Return positive-class scores for ROC (predict_proba or decision_function)."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    s = model.decision_function(X)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def load():
    df = pd.read_csv(ROOT / "data" / "M7_Data.csv").drop(columns=["ID"], errors="ignore")
    y = (df["TARGET"].astype(str).str.upper().str[0] == "Y").astype(int)
    X = df.drop(columns=["TARGET"])
    X = pd.get_dummies(X, columns=list(X.select_dtypes(include="object").columns),
                       drop_first=True, dtype=int)
    return X.fillna(X.median(numeric_only=True)), y


def main():
    X, y = load()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RNG)
    sc = StandardScaler().fit(X_tr)
    X_tr = pd.DataFrame(sc.transform(X_tr), columns=X.columns, index=X_tr.index)
    X_te = pd.DataFrame(sc.transform(X_te), columns=X.columns, index=X_te.index)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN (k=15)": KNeighborsClassifier(n_neighbors=15),
        "SVM (RBF)": SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=RNG),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=RNG, n_jobs=-1),
    }
    rows, roc_store, fitted = [], {}, {}
    for name, m in models.items():
        m.fit(X_tr, y_tr)
        proba = scores(m, X_te); pred = m.predict(X_te)
        roc_store[name] = roc_curve(y_te, proba)
        rows.append({"Model": name, "Accuracy": accuracy_score(y_te, pred),
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred, zero_division=0),
                     "F1": f1_score(y_te, pred, zero_division=0),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
        fitted[name] = m

    res = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    (ROOT / "reports").mkdir(exist_ok=True)
    cols = ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    fmt = lambda v: v if isinstance(v, str) else f"{v:.3f}"
    lines = ["# Test-set model comparison", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"]*len(cols)) + "|"]
    for _, r in res.iterrows():
        lines.append("| " + " | ".join(fmt(r[c]) for c in cols) + " |")
    (ROOT / "reports" / "model_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    best = res.iloc[0]["Model"]
    plt.figure(figsize=(7, 6))
    for name, (fpr, tpr, _) in roc_store.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_te, scores(fitted[name], X_te)):.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — Insurance Cross-Sell"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(FIG / "roc_curves.png", dpi=120); plt.close()

    ConfusionMatrixDisplay(confusion_matrix(y_te, fitted[best].predict(X_te)),
                           display_labels=["No", "Cross-sell"]).plot(cmap="Purples", colorbar=False)
    plt.title(f"Confusion Matrix — {best} (test)")
    plt.tight_layout(); plt.savefig(FIG / "confusion_matrix_best.png", dpi=120); plt.close()

    imp = pd.Series(fitted["Random Forest"].feature_importances_, index=X.columns).sort_values().tail(12)
    plt.figure(figsize=(8, 6)); imp.plot.barh(color="#6a1b9a")
    plt.title("Random Forest — Feature Importance"); plt.xlabel("Importance")
    plt.tight_layout(); plt.savefig(FIG / "feature_importance_rf.png", dpi=120); plt.close()
    print(f"Best by ROC-AUC: {best}\nWritten to {ROOT/'reports'}")


if __name__ == "__main__":
    main()
