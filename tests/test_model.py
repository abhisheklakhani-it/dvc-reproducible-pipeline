import numpy as np
import pandas as pd
import pytest

from gridload.evaluate import evaluate, scores
from gridload.train import train

TRAIN_PARAMS = {
    "learning_rate": 0.1,
    "num_leaves": 7,
    "min_child_samples": 5,
    "n_estimators": 50,
    "early_stopping_rounds": 10,
    "seed": 42,
}


def toy_features(n=600):
    rng = np.random.default_rng(1)
    hour = np.tile(np.arange(24), n // 24)
    x = rng.normal(size=len(hour))
    load = 50_000 + 5_000 * np.sin(hour / 24 * 2 * np.pi) + 1_000 * x
    split = np.where(np.arange(len(hour)) < len(hour) * 0.7, "train", "val")
    idx = pd.date_range("2025-01-01", periods=len(hour), freq="h", tz="UTC")
    return pd.DataFrame({"hour": hour, "x": x, "load_mw": load, "split": split}, index=idx)


def test_scores():
    actual = pd.Series([100.0, 200.0])
    s = scores(actual, pd.Series([110.0, 180.0]))
    assert s == {"mae_mw": 15.0, "rmse_mw": pytest.approx(15.8, abs=0.05), "mape_pct": 10.0}


def test_evaluate_reports_skill_against_each_baseline():
    idx = pd.RangeIndex(4)
    feats = pd.DataFrame({"load_mw": [100.0] * 4, "split": ["val", "val", "test", "test"]}, idx)
    pred = pd.Series([101.0, 99.0, 102.0, 98.0], idx)
    naive = pd.Series([110.0, 90.0, 104.0, 96.0], idx)
    out = evaluate(feats, pred, {"naive_week": naive})
    assert out["val"]["model"]["mae_mw"] == 1.0
    assert out["val"]["skill_vs_naive_week"] == 0.9
    assert out["test"]["skill_vs_naive_week"] == 0.5


def test_training_learns_and_is_deterministic():
    feats = toy_features()
    a, b = train(feats, TRAIN_PARAMS), train(feats, TRAIN_PARAMS)
    va = feats[feats["split"] == "val"]
    pa, pb = a.predict(va[["hour", "x"]]), b.predict(va[["hour", "x"]])
    np.testing.assert_array_equal(pa, pb)
    model_mae = np.abs(pa - va["load_mw"]).mean()
    mean_mae = np.abs(va["load_mw"].mean() - va["load_mw"]).mean()
    assert model_mae < 0.5 * mean_mae
