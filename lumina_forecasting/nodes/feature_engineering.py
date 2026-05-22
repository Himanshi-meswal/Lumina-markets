"""Feature-engineering node (shared backbone).

Builds the feature matrix every forecasting branch consumes:
  - calendar      : day-of-week, month, week, cyclical day-of-year
  - price / promo : discount depth, promo-type one-hots
  - lags/rolling  : own-series memory at multiple horizons
  - cross-product : bounded substitution + complementarity signals (4.2)

The cross-product features are deliberately O(1) per SKU, not O(n^2):
  * competitor_promo_pressure  -> share of same-subcategory rivals on promo
  * sub_units_lag1             -> lagged subcategory demand (anchor momentum)

Frequency features are added upstream by demand_pattern_agent, so this node
expects to receive the panel that already carries them.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config


def _calendar(df: pd.DataFrame) -> pd.DataFrame:
    df["dow"] = df.Date.dt.dayofweek
    df["month"] = df.Date.dt.month
    df["weekofyear"] = df.Date.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df.dow >= 5).astype(int)
    doy = df.Date.dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def _price_promo(df: pd.DataFrame) -> pd.DataFrame:
    df["is_promo"] = (df.Promotion_Type != "No_Promo").astype(int)
    df["discount_depth"] = (1 - df.Actual_Selling_Price / df.Base_Price).clip(lower=0)
    for p in ["TPR", "BOGO", "Clearance", "Seasonal"]:
        df[f"promo_{p}"] = (df.Promotion_Type == p).astype(int)
    return df


def _lags_rolling(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby(["Store_ID", "SKU_ID"])
    for lag in config.LAGS:
        df[f"lag_{lag}"] = grp["Units_Sold"].shift(lag)
    shifted = grp["Units_Sold"].shift(1)
    keys = [df.Store_ID, df.SKU_ID]
    for win in config.ROLL_WINDOWS:
        df[f"roll_mean_{win}"] = shifted.groupby(keys).transform(
            lambda s: s.rolling(win, min_periods=1).mean())
        df[f"roll_std_{win}"] = shifted.groupby(keys).transform(
            lambda s: s.rolling(win, min_periods=1).std())

    def days_since(s: pd.Series) -> pd.Series:
        arr = s.to_numpy()
        out = np.empty(arr.size)
        cnt = 999
        for i, x in enumerate(arr):
            out[i] = cnt
            cnt = 0 if x > 0 else cnt + 1
        return pd.Series(out, index=s.index)

    df["days_since_sale"] = grp["Units_Sold"].transform(days_since)
    return df


def _cross_product(df: pd.DataFrame) -> pd.DataFrame:
    sub_day = (
        df.groupby(["Store_ID", "Sub_Category", "Date"])
        .agg(sub_promo_share=("is_promo", "mean"),
             sub_n=("SKU_ID", "count"),
             sub_units=("Units_Sold", "sum"))
        .reset_index()
    )
    df = df.merge(sub_day, on=["Store_ID", "Sub_Category", "Date"], how="left")
    # substitution: rival promo pressure, excluding the SKU's own contribution
    df["competitor_promo_pressure"] = (
        (df.sub_promo_share * df.sub_n - df.is_promo) / (df.sub_n - 1).clip(lower=1)
    )
    # complementarity: lagged subcategory momentum (anchor effect proxy)
    df = df.sort_values(["Store_ID", "Sub_Category", "Date"])
    df["sub_units_lag1"] = df.groupby(["Store_ID", "Sub_Category"])["sub_units"].shift(1)
    df = df.sort_values(["Store_ID", "SKU_ID", "Date"]).reset_index(drop=True)
    return df


def _cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    for c in config.CATEGORICAL_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    if "Perishability_Flag" in df.columns:
        df["Perishability_Flag"] = df.Perishability_Flag.astype(int)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Columns the model is allowed to see."""
    return [c for c in df.columns if c not in config.NON_FEATURE_COLS]


def run(panel: pd.DataFrame) -> pd.DataFrame:
    """Entry point: panel (with freq features) -> full feature matrix."""
    df = panel.sort_values(["Store_ID", "SKU_ID", "Date"]).copy()
    df = _calendar(df)
    df = _price_promo(df)
    df = _lags_rolling(df)
    df = _cross_product(df)
    df = _cast_categoricals(df)
    return df


if __name__ == "__main__":
    from . import data_agent, demand_pattern_agent
    d = data_agent.run(persist=False)
    dp = demand_pattern_agent.run(d["panel"], d["tables"]["sales"], d["tables"]["product"])
    feats = run(dp["panel"])
    print(f"Feature matrix: {feats.shape[0]:,} rows x {feats.shape[1]} cols")
    print("Model features:", len(feature_columns(feats)))
