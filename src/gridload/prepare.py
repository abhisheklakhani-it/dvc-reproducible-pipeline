"""Stage 1: raw CSVs -> one clean hourly table (UTC) with load and national weather."""

from __future__ import annotations

import pandas as pd

from gridload.config import PREPARED, RAW
from gridload.geo import weights

MAX_GAP_H = 3  # interpolate holes up to 3 hours; longer ones stay missing


def national_weather(weather: pd.DataFrame) -> pd.DataFrame:
    """Population-weighted mean of each weather variable over the 16 states."""
    w = weather["state"].map(weights())
    cols = [c for c in weather.columns if c not in ("time", "state")]
    weighted = weather[cols].mul(w, axis=0)
    weighted["time"] = weather["time"]
    out = weighted.groupby("time")[cols].sum()
    return out.rename(columns={"temperature_2m": "temp_c", "shortwave_radiation": "radiation_wm2"})


def clean_load(load: pd.Series) -> pd.Series:
    """Hourly grid with short gaps filled; zero or negative readings count as missing."""
    load = load.where(load > 0)
    full = pd.date_range(load.index.min(), load.index.max(), freq="h", tz="UTC")
    load = load.reindex(full)
    # interpolate(limit=n) would fill the first n hours of a long gap too; measure each gap instead
    missing = load.isna()
    gap_len = missing.groupby((~missing).cumsum()).transform("sum")
    filled = load.interpolate(limit_area="inside")
    return filled.where(~(missing & (gap_len > MAX_GAP_H)))


def main() -> None:
    load = pd.read_csv(RAW / "load.csv", index_col="time", parse_dates=["time"])["load_mw"]
    weather = pd.read_csv(RAW / "weather.csv", parse_dates=["time"])
    tso = pd.read_csv(RAW / "tso_forecast.csv", index_col="time", parse_dates=["time"])

    raw_missing = int(load.le(0).sum() + load.isna().sum())
    load = clean_load(load)
    df = load.to_frame("load_mw").join(national_weather(weather), how="left")
    df = df.join(tso["tso_forecast_mw"].where(tso["tso_forecast_mw"] > 0), how="left")
    df.index.name = "time"

    PREPARED.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(PREPARED)
    print(
        f"prepared {len(df):,} hours ({df.index.min()} -> {df.index.max()}), "
        f"{raw_missing} bad raw readings, {int(df['load_mw'].isna().sum())} still missing"
    )


if __name__ == "__main__":
    main()
