import numpy as np
import pandas as pd
import pytest

from gridload.featurize import build_features, calendar_features, holiday_share

PARAMS = {
    "split": {
        "train_start": "2023-12-01",
        "val_start": "2024-01-01",
        "test_start": "2024-01-15",
        "end": "2024-01-31",
    },
    "features": {"lags_h": [48, 168], "weather": True, "holidays": True},
}


def hourly(start="2023-11-20", end="2024-02-01"):
    idx = pd.date_range(start, end, freq="h", tz="UTC", inclusive="left")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "load_mw": 50_000 + rng.normal(0, 1000, len(idx)),
            "temp_c": rng.normal(5, 3, len(idx)),
            "radiation_wm2": rng.uniform(0, 300, len(idx)),
        },
        index=idx,
    )


def test_holiday_share_national_regional_and_normal_day():
    days = pd.DatetimeIndex(["2025-12-25", "2025-06-19", "2025-06-18"])
    share = holiday_share(days)
    assert share.iloc[0] == pytest.approx(1.0)  # Christmas: everyone
    assert 0.5 < share.iloc[1] < 0.75  # Corpus Christi: BW, BY, HE, NW, RP, SL
    assert share.iloc[2] == 0.0


def test_bridge_day_after_ascension():
    idx = pd.date_range("2025-05-28", "2025-06-02", freq="h", tz="Europe/Berlin").tz_convert("UTC")
    f = calendar_features(idx, with_holidays=True)
    local_day = idx.tz_convert("Europe/Berlin").strftime("%Y-%m-%d")
    by_day = f.groupby(local_day)["bridge_day"].max()
    assert by_day["2025-05-30"] == 1  # Friday after Ascension Thursday
    assert by_day["2025-05-28"] == 0


def test_lags_are_exact_shifts_and_rows_with_gaps_are_dropped():
    df = hourly()
    feats = build_features(df, PARAMS)
    t = feats.index[100]
    assert feats.loc[t, "load_lag_48h"] == df.loc[t - pd.Timedelta(hours=48), "load_mw"]
    assert feats.loc[t, "load_lag_168h"] == df.loc[t - pd.Timedelta(hours=168), "load_mw"]
    assert not feats.isna().any().any()


def test_lag_shorter_than_forecast_gap_is_rejected():
    params = {**PARAMS, "features": {**PARAMS["features"], "lags_h": [24, 168]}}
    with pytest.raises(ValueError, match="not known at forecast time"):
        build_features(hourly(), params)


def test_split_follows_local_dates():
    feats = build_features(hourly(), PARAMS)
    # 23:00 UTC on 31 Dec is already midnight of 1 January in Berlin
    assert feats.loc["2023-12-31 23:00+00:00", "split"] == "val"
    assert feats.loc["2023-12-31 22:00+00:00", "split"] == "train"
    assert feats.loc["2024-01-20 12:00+00:00", "split"] == "test"
    assert feats.index.tz_convert("Europe/Berlin").max() < pd.Timestamp(
        "2024-02-01", tz="Europe/Berlin"
    )


def test_feature_switches_remove_columns():
    params = {**PARAMS, "features": {"lags_h": [48], "weather": False, "holidays": False}}
    cols = set(build_features(hourly(), params).columns)
    assert not {"temp_c", "holiday_share", "load_lag_168h"} & cols
    assert {"hour", "load_lag_48h", "load_mean_7d", "load_mw", "split"} <= cols
