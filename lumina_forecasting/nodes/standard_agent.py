"""Standard forecasting agent — the global quantile model.

This is the workhorse the whole tree shares. One LightGBM model per quantile,
trained across ALL series (global model) so sparse long-tail SKUs borrow shape
from dense ones. Smooth / Erratic / Intermittent / Lumpy classes all route here;
the quantile (pinball) loss handles the 90%-zero sparsity natively and yields
the P10/P50/P90 distribution that inventory planning consumes.

Halo and NPD agents import `train_global` / `predict` from here rather than
re-implementing the model — they only differ in which features they emphasise
or how they assemble the training frame.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from .. import config
from ..nodes import feature_engineering


def train_test_split(features: pd.DataFrame, horizon_days: int | None = None):
    """Time-based split: last `horizon_days` are the holdout."""
    h = horizon_days or config.TEST_HORIZON_DAYS
    cutoff = features.Date.max() - pd.Timedelta(days=h)
    train = features[features.Date <= cutoff].dropna(subset=[f"lag_{max(config.LAGS)}"])
    test = features[features.Date > cutoff]
    return train, test, cutoff


def train_global(train: pd.DataFrame,
                 feat_cols: list[str],
                 quantiles: list[float] | None = None) -> dict:
    """Fit one LightGBM regressor per quantile. Returns {q: model}."""
    qs = quantiles or config.QUANTILES
    cat = [c for c in config.CATEGORICAL_COLS if c in feat_cols]
    X, y = train[feat_cols], train["Units_Sold"]
    models = {}
    for q in qs:
        m = lgb.LGBMRegressor(alpha=q, **config.LGBM.as_dict())
        m.fit(X, y, categorical_feature=cat)
        models[q] = m
    return models


def predict(models: dict, frame: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Predict every quantile; enforce monotonicity (P10<=P50<=P90)."""
    out = frame[["Date", "Store_ID", "SKU_ID", "Units_Sold"]].copy()
    qs = sorted(models)
    preds = np.column_stack([np.clip(models[q].predict(frame[feat_cols]), 0, None) for q in qs])
    preds = np.sort(preds, axis=1)  # guarantee non-crossing quantiles
    for i, q in enumerate(qs):
        out[f"P{int(q * 100)}"] = preds[:, i]
    return out


def feature_importance(models: dict, feat_cols: list[str], q: float = 0.5) -> pd.Series:
    return pd.Series(models[q].feature_importances_, index=feat_cols).sort_values(ascending=False)


def run(features: pd.DataFrame,
        sku_subset: list[str] | None = None,
        horizon_days: int | None = None) -> dict:
    """Entry point.

    Parameters
    ----------
    features   : full feature matrix from feature_engineering.run()
    sku_subset : optional list of SKU_IDs to *predict* for (model still trains
                 globally on all data for pattern sharing). Used by the
                 orchestrator to keep each branch's outputs separate.

    Returns dict: models, predictions, importance, cutoff.
    """
    feat_cols = feature_engineering.feature_columns(features)
    train, test, cutoff = train_test_split(features, horizon_days)
    models = train_global(train, feat_cols)

    target = test
    if sku_subset is not None:
        target = test[test.SKU_ID.isin(sku_subset)]
    predictions = predict(models, target, feat_cols)
    imp = feature_importance(models, feat_cols)
    return {"models": models, "predictions": predictions,
            "importance": imp, "cutoff": cutoff, "feat_cols": feat_cols}


if __name__ == "__main__":
    from . import data_agent, demand_pattern_agent
    d = data_agent.run(persist=False)
    dp = demand_pattern_agent.run(d["panel"], d["tables"]["sales"], d["tables"]["product"])
    feats = feature_engineering.run(dp["panel"])
    res = run(feats)
    p = res["predictions"]
    print(f"Predicted {len(p):,} rows. Top features:")
    print(res["importance"].head(8).to_string())
