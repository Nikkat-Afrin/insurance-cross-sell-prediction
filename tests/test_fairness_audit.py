"""Tests for the fairness/calibration audit — synthetic cases with known bias."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fairness_audit import (age_band, audit, group_metrics, load,  # noqa: E402
                            reliability_table)


def test_age_band_covers_all_ages():
    ages = pd.Series([5, 25, 26, 35, 36, 50, 51, 102])
    bands = age_band(ages)
    assert bands.notna().all()
    assert list(bands) == ["<=25", "<=25", "26-35", "26-35",
                           "36-50", "36-50", "51+", "51+"]


def test_group_metrics_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.2, 0.8, 0.9])
    pred = (proba >= 0.5).astype(int)
    m = group_metrics(y, proba, pred)
    assert m["tpr"] == 1.0 and m["fpr"] == 0.0 and m["auc"] == 1.0
    assert m["selection_rate"] == 0.5


def test_audit_flags_known_bias():
    """Group B gets systematically lower scores -> must be parity-flagged."""
    rng = np.random.default_rng(1)
    n = 4000
    groups = pd.Series(np.where(rng.random(n) < 0.5, "A", "B"))
    y = pd.Series(rng.integers(0, 2, n))
    proba = np.where(groups == "A",
                     np.clip(0.5 + y * 0.3 + rng.normal(0, 0.1, n), 0, 1),
                     np.clip(0.2 + y * 0.2 + rng.normal(0, 0.1, n), 0, 1))
    report = audit(y, proba, groups)
    assert "B" in report["parity"]["flagged"]
    assert report["parity"]["reference_group"] == "A"
    assert report["parity"]["ratios"]["A"] == 1.0


def test_audit_no_flags_when_groups_identical():
    rng = np.random.default_rng(2)
    n = 4000
    groups = pd.Series(np.where(rng.random(n) < 0.5, "A", "B"))
    y = pd.Series(rng.integers(0, 2, n))
    proba = np.clip(0.3 + y * 0.4 + rng.normal(0, 0.05, n), 0, 1)
    report = audit(y, proba, groups)
    assert report["parity"]["flagged"] == []


def test_reliability_table_well_calibrated():
    """Scores drawn as true probabilities must sit on the diagonal."""
    rng = np.random.default_rng(3)
    proba = rng.random(20000)
    y = (rng.random(20000) < proba).astype(int)
    table = reliability_table(y, proba)
    deviation = (table["mean_predicted"] - table["observed_rate"]).abs()
    assert deviation.max() < 0.05


def test_real_data_loads_with_age_intact():
    X, y, age = load()
    assert len(X) == len(y) == len(age) > 10000
    assert "TARGET" not in X.columns
    assert age.between(0, 120).all()
