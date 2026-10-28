"""Download the raw data once; the result is versioned with `dvc add data/raw`.

- Load: SMARD (Bundesnetzagentur), series 410 "Netzlast", hourly, MWh per hour.
- Weather: Open-Meteo historical archive, hourly 2 m temperature and solar
  radiation for one city per federal state.

The pipeline never calls these APIs, so `dvc repro` does not depend on them
being up or returning the same numbers later.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from gridload.geo import STATES

SMARD = "https://www.smard.de/app/chart_data/410/DE"
OPEN_METEO = "https://archive-api.open-meteo.com/v1/archive"
WEATHER_VARS = ["temperature_2m", "shortwave_radiation"]


def _get(url: str, **params) -> dict:
    for attempt in range(5):
        resp = requests.get(url, params=params, timeout=60)
        if resp.status_code == 200:
            return resp.json()
        time.sleep(2**attempt)
    resp.raise_for_status()
    raise RuntimeError(url)


def download_load(start: str, end: str) -> pd.DataFrame:
    """Hourly grid load in MW (MWh per hour) between start and end (UTC, inclusive)."""
    lo = pd.Timestamp(start, tz="UTC")
    hi = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    weeks = _get(f"{SMARD}/index_hour.json")["timestamps"]
    # each file holds one week starting at the timestamp; keep weeks that overlap the range
    week_ms = 7 * 24 * 3600 * 1000
    wanted = [t for t in weeks if lo.value // 10**6 - week_ms < t <= hi.value // 10**6]
    rows = []
    for t in wanted:
        rows += _get(f"{SMARD}/410_DE_hour_{t}.json")["series"]
    df = pd.DataFrame(rows, columns=["ms", "load_mw"])
    df["time"] = pd.to_datetime(df["ms"], unit="ms", utc=True)
    df = df.drop(columns="ms").drop_duplicates("time").set_index("time").sort_index()
    return df.loc[lo:hi]


def download_weather(start: str, end: str) -> pd.DataFrame:
    """Hourly weather per state in long format: time, state, temperature_2m, shortwave_radiation."""
    frames = []
    for code, (_, lat, lon) in STATES.items():
        data = _get(
            OPEN_METEO,
            latitude=lat,
            longitude=lon,
            start_date=start,
            end_date=end,
            hourly=",".join(WEATHER_VARS),
            timezone="UTC",
        )["hourly"]
        df = pd.DataFrame(data)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df.insert(1, "state", code)
        frames.append(df)
        time.sleep(1)  # be polite to the free API
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-12-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", type=Path, default=Path("data/raw"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    load = download_load(args.start, args.end)
    load.to_csv(args.out / "load.csv")
    print(f"load: {len(load):,} hours, {load.index.min()} -> {load.index.max()}")

    weather = download_weather(args.start, args.end)
    weather.to_csv(args.out / "weather.csv", index=False)
    print(f"weather: {len(weather):,} rows for {weather['state'].nunique()} states")


if __name__ == "__main__":
    main()
