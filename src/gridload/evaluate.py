"""Stage 4: score the model and a seasonal-naive baseline on val and test.

Two baselines, scored on exactly the same hours as the model:
- naive_week: load at the same hour one week earlier, the usual yardstick.
- tso_day_ahead: the grid operators' official day-ahead forecast (SMARD 411).
  Not a fair fight: the model sees observed weather, the operators only had a
  weather forecast, so treat the gap to it as an upper bound.
Writes metrics for `dvc metrics` and data/images for `dvc plots`.
"""

from __future__ import annotations

import json

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd

from gridload.config import FEATURES, MODEL, PREPARED, REPORTS, TARGET, TZ

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PLOTS = REPORTS / "plots"
COLORS = {"model": "#1f77b4", "tso_day_ahead": "#2ca02c", "naive_week": "#ff7f0e"}
FIGURES = REPORTS / "figures"


def scores(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    err = pred - actual
    return {
        "mae_mw": round(float(err.abs().mean()), 1),
        "rmse_mw": round(float(np.sqrt((err**2).mean())), 1),
        "mape_pct": round(float((err.abs() / actual).mean() * 100), 3),
    }


def evaluate(feats: pd.DataFrame, pred: pd.Series, baselines: dict[str, pd.Series]) -> dict:
    out = {}
    for split in ("val", "test"):
        m = feats["split"] == split
        actual = feats.loc[m, TARGET]
        res = {"model": scores(actual, pred[m])}
        for name, base in baselines.items():
            res[name] = scores(actual, base[m])
            res[f"skill_vs_{name}"] = round(1 - res["model"]["mae_mw"] / res[name]["mae_mw"], 4)
        out[split] = res
    return out


def write_plots(
    feats: pd.DataFrame, pred: pd.Series, naive: pd.Series, tso: pd.Series, booster
) -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    test = feats[feats["split"] == "test"]
    local = test.index.tz_convert(TZ)

    # one winter week of the test year, including a weekend
    start = pd.Timestamp("2025-02-03", tz=TZ)
    week = (local >= start) & (local < start + pd.Timedelta(days=7))
    wk = pd.DataFrame(
        {
            "time": local[week].strftime("%Y-%m-%d %H:%M"),
            "actual_mw": test.loc[week, TARGET].round(0),
            "forecast_mw": pred[test.index][week].round(0),
            "naive_week_mw": naive[test.index][week].round(0),
            "tso_day_ahead_mw": tso[test.index][week].round(0),
        }
    )
    wk.to_csv(PLOTS / "forecast_week.csv", index=False)

    by_hour = (
        pd.DataFrame(
            {
                "hour": local.hour,
                "model": (pred[test.index] - test[TARGET]).abs(),
                "naive_week": (naive[test.index] - test[TARGET]).abs(),
                "tso_day_ahead": (tso[test.index] - test[TARGET]).abs(),
            }
        )
        .groupby("hour")[["model", "naive_week", "tso_day_ahead"]]
        .mean()
        .round(1)
    )
    by_hour.reset_index().to_csv(PLOTS / "mae_by_hour.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 4.4))
    x = pd.to_datetime(wk["time"])
    ax.plot(x, wk["actual_mw"] / 1000, label="actual", color="black", lw=1.6)
    ax.plot(x, wk["forecast_mw"] / 1000, label="LightGBM forecast", color=COLORS["model"], lw=1.3)
    ax.plot(
        x,
        wk["tso_day_ahead_mw"] / 1000,
        label="official day-ahead",
        color=COLORS["tso_day_ahead"],
        lw=1,
    )
    ax.plot(
        x,
        wk["naive_week_mw"] / 1000,
        label="same hour last week",
        color=COLORS["naive_week"],
        ls="--",
    )
    ax.set_ylabel("German grid load (GW)")
    ax.set_title("Test year: week of 3 Feb 2025")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "forecast_week.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = {
        "model": "LightGBM",
        "tso_day_ahead": "official day-ahead",
        "naive_week": "same hour last week",
    }
    for col, label in labels.items():
        ax.plot(by_hour.index, by_hour[col], marker="o", label=label, color=COLORS[col])
    ax.set_xlabel("hour of day (local time)")
    ax.set_ylabel("MAE (MW), test year")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "mae_by_hour.png", dpi=130)
    plt.close(fig)

    gain = pd.Series(booster.feature_importance("gain"), index=booster.feature_name()).sort_values()
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(gain) + 1))
    ax.barh(gain.index, gain / gain.sum() * 100, color=COLORS["model"])
    ax.set_xlabel("share of total gain (%)")
    ax.set_title("What the model relies on")
    fig.tight_layout()
    fig.savefig(FIGURES / "feature_importance.png", dpi=130)
    plt.close(fig)


def main() -> None:
    feats = pd.read_parquet(FEATURES)
    prepared = pd.read_parquet(PREPARED)
    booster = lgb.Booster(model_file=str(MODEL))

    pred = pd.Series(booster.predict(feats[booster.feature_name()]), index=feats.index)
    naive = prepared[TARGET].shift(168).reindex(feats.index)
    tso = prepared["tso_forecast_mw"].reindex(feats.index)
    keep = naive.notna() & tso.notna()
    feats, pred, naive, tso = feats[keep], pred[keep], naive[keep], tso[keep]

    metrics = evaluate(feats, pred, {"naive_week": naive, "tso_day_ahead": tso})
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_plots(feats, pred, naive, tso, booster)
    for split, m in metrics.items():
        print(
            f"{split}: model MAE {m['model']['mae_mw']:.0f} MW ({m['model']['mape_pct']:.2f}%), "
            f"official {m['tso_day_ahead']['mae_mw']:.0f} MW, "
            f"last week {m['naive_week']['mae_mw']:.0f} MW"
        )


if __name__ == "__main__":
    main()
