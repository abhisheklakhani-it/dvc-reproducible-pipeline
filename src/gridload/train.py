"""Stage 3: fit LightGBM on train, stop early on val, save the booster."""

from __future__ import annotations

import json

import lightgbm as lgb
import pandas as pd

from gridload.config import FEATURES, MODEL, REPORTS, TARGET, load_params


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in (TARGET, "split")]


def train(feats: pd.DataFrame, tp: dict) -> lgb.Booster:
    cols = feature_columns(feats)
    tr, va = feats[feats["split"] == "train"], feats[feats["split"] == "val"]
    dtrain = lgb.Dataset(tr[cols], tr[TARGET])
    dval = lgb.Dataset(va[cols], va[TARGET], reference=dtrain)
    lgb_params = {
        "objective": "l1",  # MAE: the error unit grid operators talk about, robust to outliers
        "learning_rate": tp["learning_rate"],
        "num_leaves": tp["num_leaves"],
        "min_child_samples": tp["min_child_samples"],
        "seed": tp["seed"],
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    return lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=tp["n_estimators"],
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(tp["early_stopping_rounds"], verbose=False)],
    )


def main() -> None:
    params = load_params()
    feats = pd.read_parquet(FEATURES)
    booster = train(feats, params["train"])

    MODEL.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(MODEL, num_iteration=booster.best_iteration)
    REPORTS.mkdir(exist_ok=True)
    info = {"best_iteration": booster.best_iteration, "n_features": booster.num_feature()}
    (REPORTS / "train.json").write_text(json.dumps(info, indent=2) + "\n")
    print(f"best iteration {info['best_iteration']}, {info['n_features']} features")


if __name__ == "__main__":
    main()
