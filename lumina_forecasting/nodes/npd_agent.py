"""NPD (new-product) agent — cold-start forecasting (4.6).

A brand-new SKU has no history, so own-lag features are null. Two mechanisms:

1. **Global static features**: the shared model already keys off Category,
   Brand_Tier, Perishability_Flag and store attributes, so it can emit a day-1
   forecast from attributes alone.

2. **Analog borrowing**: find the k most similar existing SKUs (same
   sub-category / brand tier / perishability) and use the mean of their
   early-life demand as a prior / fallback when the model is unsure.

This node simulates cold start by treating a chosen SKU as new (masking its
history) so the approach is demonstrable on the synthetic data. In production
`new_sku_attrs` would describe the genuinely new item.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from ..nodes import standard_agent, feature_engineering


def find_analogs(product: pd.DataFrame, attrs: dict, k: int = 5) -> list[str]:
    """Rank existing SKUs by attribute match to the new item."""
    p = product.copy()
    score = pd.Series(0, index=p.index)
    for col, weight in [("Sub_Category", 3), ("Category", 2),
                        ("Brand_Tier", 2), ("Perishability_Flag", 1)]:
        if col in attrs and col in p.columns:
            score += weight * (p[col] == attrs[col]).astype(int)
    p = p.assign(_score=score)
    if "SKU_ID" in attrs:
        p = p[p.SKU_ID != attrs["SKU_ID"]]
    return p.sort_values("_score", ascending=False).head(k).SKU_ID.tolist()


def analog_prior(sales: pd.DataFrame, analog_skus: list[str],
                 launch_days: int = 28) -> float:
    """Mean early-life daily demand across analog SKUs (per store)."""
    sub = sales[sales.SKU_ID.isin(analog_skus)].copy()
    if sub.empty:
        return 0.0
    sub = sub.sort_values(["SKU_ID", "Store_ID", "Date"])
    early = sub.groupby(["SKU_ID", "Store_ID"]).head(launch_days)
    return float(early.Units_Sold.mean())


def run(features: pd.DataFrame,
        sales: pd.DataFrame,
        product: pd.DataFrame,
        new_sku_attrs: dict,
        models: dict | None = None,
        horizon_days: int | None = None,
        k: int = 5) -> dict:
    """Entry point.

    Parameters
    ----------
    new_sku_attrs : dict describing the new item, e.g.
        {"SKU_ID": "SKU0034", "Sub_Category": "Strawberries",
         "Category": "Produce", "Brand_Tier": "Premium", "Perishability_Flag": 1}
        (SKU_ID optional; if present it is excluded from its own analogs.)

    Returns dict: analogs, analog_prior, model_forecast (if the SKU exists in
    the holdout for demonstration), blended_day1.
    """
    feat_cols = feature_engineering.feature_columns(features)
    analogs = find_analogs(product, new_sku_attrs, k=k)
    prior = analog_prior(sales, analogs)

    model_forecast = None
    target_sku = new_sku_attrs.get("SKU_ID")
    if models is None:
        train, test, cutoff = standard_agent.train_test_split(features, horizon_days)
        models = standard_agent.train_global(train, feat_cols)
    else:
        _, test, cutoff = standard_agent.train_test_split(features, horizon_days)

    if target_sku is not None and target_sku in set(test.SKU_ID):
        preds = standard_agent.predict(models, test[test.SKU_ID == target_sku], feat_cols)
        model_forecast = float(preds["P50"].mean())

    # day-1 blend: lean on analogs when the model has no own-history signal
    blended = model_forecast if model_forecast is not None else prior
    blended = 0.5 * blended + 0.5 * prior if model_forecast is not None else prior

    return {"analogs": analogs, "analog_prior": round(prior, 2),
            "model_forecast_p50": None if model_forecast is None else round(model_forecast, 2),
            "blended_day1": round(float(blended), 2), "cutoff": cutoff}


if __name__ == "__main__":
    from . import data_agent, demand_pattern_agent
    d = data_agent.run(persist=False)
    dp = demand_pattern_agent.run(d["panel"], d["tables"]["sales"], d["tables"]["product"])
    feats = feature_engineering.run(dp["panel"])
    attrs = {"SKU_ID": "SKU0034", "Sub_Category": "Strawberries",
             "Category": "Produce", "Brand_Tier": "Premium", "Perishability_Flag": 1}
    res = run(feats, d["tables"]["sales"], d["tables"]["product"], attrs)
    print("Analogs:", res["analogs"])
    print("Analog prior:", res["analog_prior"], "| blended day-1:", res["blended_day1"])
