"""Stage 2: hourly table -> model features, target and a train/val/test label.

The forecast is day-ahead: on day D-1 around noon, predict all 24 hours of
day D (the German day-ahead market closes at 12:00). The last complete hour
known then is 11:00 on D-1, which is up to 36 hours before the last target
hour (23:00 on D). Every load lag must therefore be at least 36 hours.
Weather uses the observed values as a stand-in for a weather forecast.
"""

from __future__ import annotations

from typing import Any

import holidays
import numpy as np
import pandas as pd

from gridload.config import FEATURES, PREPARED, TARGET, TZ, load_params
from gridload.geo import weights

MIN_LAG_H = 36


def holiday_share(dates: pd.DatetimeIndex) -> pd.Series:
    """Share of the German population with a public holiday on each local date."""
    years = sorted(set(dates.year))
    share = pd.Series(0.0, index=dates)
    for code, w in weights().items():
        cal = holidays.country_holidays("DE", subdiv=code, years=years)
        share += np.array([d in cal for d in dates.date]) * w
    return share


def calendar_features(index: pd.DatetimeIndex, with_holidays: bool) -> pd.DataFrame:
    local = index.tz_convert(TZ)
    f = pd.DataFrame(index=index)
    f["hour"] = local.hour
    f["dow"] = local.day_of_week
    f["month"] = local.month
    f["doy"] = local.day_of_year
    if not with_holidays:
        return f

    days = pd.DatetimeIndex(sorted(set(local.normalize().tz_localize(None))))
    share = holiday_share(days)
    weekday = days.day_of_week < 5
    prev_hol = share.shift(1, fill_value=0).to_numpy() > 0.5
    next_hol = share.shift(-1, fill_value=0).to_numpy() > 0.5
    # a working day squeezed between a holiday and the weekend: many people take it off
    bridge = (
        weekday
        & (share.to_numpy() < 0.5)
        & (((days.day_of_week == 4) & prev_hol) | ((days.day_of_week == 0) & next_hol))
    )
    xmas = ((days.month == 12) & (days.day >= 24)) | ((days.month == 1) & (days.day == 1))
    daily = pd.DataFrame(
        {"holiday_share": share.to_numpy(), "bridge_day": bridge, "christmas_week": xmas},
        index=days,
    )
    per_hour = daily.reindex(local.normalize().tz_localize(None)).to_numpy()
    f[["holiday_share", "bridge_day", "christmas_week"]] = per_hour
    f[["bridge_day", "christmas_week"]] = f[["bridge_day", "christmas_week"]].astype(int)
    f["holiday_share"] = f["holiday_share"].astype(float)
    return f


def build_features(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
    fp, sp = params["features"], params["split"]
    lags = sorted(fp["lags_h"])
    if lags and lags[0] < MIN_LAG_H:
        raise ValueError(f"lag {lags[0]} h is not known at forecast time (minimum {MIN_LAG_H} h)")

    out = calendar_features(df.index, fp["holidays"])
    if fp["weather"]:
        out["temp_c"] = df["temp_c"]
        out["heating_deg"] = (15 - df["temp_c"]).clip(lower=0)
        out["cooling_deg"] = (df["temp_c"] - 22).clip(lower=0)
        out["radiation_wm2"] = df["radiation_wm2"]
    for lag in lags:
        out[f"load_lag_{lag}h"] = df[TARGET].shift(lag)
    if lags:
        out["load_mean_7d"] = df[TARGET].shift(lags[0]).rolling(168, min_periods=120).mean()
    out[TARGET] = df[TARGET]

    local_day = df.index.tz_convert(TZ).normalize().tz_localize(None)
    out["split"] = np.select(
        [local_day >= pd.Timestamp(sp["test_start"]), local_day >= pd.Timestamp(sp["val_start"])],
        ["test", "val"],
        default="train",
    )
    keep = (local_day >= pd.Timestamp(sp["train_start"])) & (local_day <= pd.Timestamp(sp["end"]))
    return out[keep].dropna()


def main() -> None:
    params = load_params()
    df = pd.read_parquet(PREPARED)
    feats = build_features(df, params)
    FEATURES.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(FEATURES)
    counts = feats["split"].value_counts().reindex(["train", "val", "test"]).to_dict()
    n_feat = feats.shape[1] - 2
    print(f"{n_feat} features, rows per split: {counts}")


if __name__ == "__main__":
    main()
