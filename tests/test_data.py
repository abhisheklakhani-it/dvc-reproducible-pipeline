import numpy as np
import pandas as pd
import pytest

from gridload.geo import STATES, weights
from gridload.prepare import clean_load, national_weather


def test_weights_cover_all_states_and_sum_to_one():
    w = weights()
    assert set(w) == set(STATES) and len(w) == 16
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["NW"] > w["HB"]  # North Rhine-Westphalia counts far more than Bremen


def test_clean_load_fills_short_gaps_only():
    idx = pd.date_range("2025-01-01", periods=12, freq="h", tz="UTC")
    s = pd.Series(np.arange(12, dtype=float) + 100, index=idx)
    s.iloc[2:4] = np.nan  # with the dropped hour below: a 3-hour gap, filled
    s.iloc[6:10] = np.nan  # 4-hour gap: too long, stays missing entirely
    s = s.drop(idx[1])  # a missing timestamp is reinserted first
    out = clean_load(s)
    assert len(out) == 12
    assert out.iloc[1:4].tolist() == [101.0, 102.0, 103.0]
    assert out.iloc[6:10].isna().all()


def test_clean_load_treats_zero_as_missing():
    idx = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    out = clean_load(pd.Series([100.0, 0.0, 120.0], index=idx))
    assert out.iloc[1] == 110.0


def test_national_weather_is_population_weighted():
    t = pd.Timestamp("2025-01-01", tz="UTC")
    rows = [
        {
            "time": t,
            "state": code,
            "temperature_2m": 10.0 if code == "BY" else 0.0,
            "shortwave_radiation": 100.0,
        }
        for code in STATES
    ]
    out = national_weather(pd.DataFrame(rows))
    assert out.loc[t, "temp_c"] == pytest.approx(10 * weights()["BY"])
    assert out.loc[t, "radiation_wm2"] == pytest.approx(100.0)
