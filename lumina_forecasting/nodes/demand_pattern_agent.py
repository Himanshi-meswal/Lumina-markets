"""Demand-pattern agent.

Two outputs, both consumed by the orchestrator's routing decision:

1. **Static classification** per SKU via Syntetos-Boylan (ADI x CV^2 quadrant).
   This is the deterministic rule the orchestrator routes on.

2. **Purchase-frequency features** computed on TRAILING windows (leakage-safe),
   added to the panel so the forecasting model can react to a SKU drifting
   between archetypes over time rather than relying on one static label.

Frequency of purchase and ADI are two views of the same quantity:
    purchase_frequency = (# non-zero days) / (# days)  ~=  1 / ADI
We expose both the scalar class and the rolling frequency signal.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config


# ----------------------------------------------------------------------------
# 1. Static Syntetos-Boylan classification
# ----------------------------------------------------------------------------
def _sba_stats(series: pd.Series) -> pd.Series:
    s = series.to_numpy()
    nz = s[s > 0]
    if nz.size == 0:
        return pd.Series({"ADI": np.inf, "CV2": np.nan, "Class": "Dead"})
    adi = s.size / nz.size
    cv2 = (nz.std() / nz.mean()) ** 2 if nz.size > 1 and nz.mean() > 0 else 0.0
    if adi < config.ADI_CUTOFF and cv2 < config.CV2_CUTOFF:
        cls = "Smooth"
    elif adi < config.ADI_CUTOFF:
        cls = "Erratic"
    elif cv2 < config.CV2_CUTOFF:
        cls = "Intermittent"
    else:
        cls = "Lumpy"
    return pd.Series({"ADI": adi, "CV2": cv2, "Class": cls})


def classify(sales: pd.DataFrame, product: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to SKU-day total demand, then classify each SKU."""
    sku_daily = sales.groupby(["SKU_ID", "Date"])["Units_Sold"].sum().reset_index()
    cls = (
        sku_daily.groupby("SKU_ID")["Units_Sold"]
        .apply(_sba_stats)
        .unstack()
        .reset_index()
        .merge(product[["SKU_ID", "Category", "Sub_Category", "Brand_Tier"]], on="SKU_ID")
    )
    cls["Branch"] = cls["Class"].map(config.CLASS_TO_BRANCH)
    return cls


# ----------------------------------------------------------------------------
# 2. Purchase-frequency features (leakage-safe trailing windows)
# ----------------------------------------------------------------------------
def add_frequency_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Per store-SKU rolling purchase frequency + acceleration.

    All windows look strictly backward (shift(1) before rolling) so a row's
    features never include its own day's sale.
    """
    df = panel.sort_values(["Store_ID", "SKU_ID", "Date"]).copy()
    sold_flag = (df["Units_Sold"] > 0).astype(float)
    grp_keys = [df.Store_ID, df.SKU_ID]

    prior = sold_flag.groupby(grp_keys).shift(1)  # exclude today
    for w in config.FREQ_WINDOWS:
        df[f"freq_{w}d"] = (
            prior.groupby(grp_keys)
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
    # acceleration: short-window freq minus long-window freq (positive = speeding up)
    if len(config.FREQ_WINDOWS) >= 2:
        short, long = sorted(config.FREQ_WINDOWS)[:2]
        df["freq_accel"] = df[f"freq_{short}d"] - df[f"freq_{long}d"]

    # expanding inter-purchase interval (running ADI proxy), leakage-safe
    def running_adi(sold: pd.Series) -> pd.Series:
        arr = sold.to_numpy()
        out = np.empty(arr.size)
        days = 0
        nz = 0
        for i, x in enumerate(arr):
            out[i] = (days / nz) if nz > 0 else np.nan  # uses history up to i-1
            days += 1
            if x > 0:
                nz += 1
        return pd.Series(out, index=sold.index)

    df["running_adi"] = sold_flag.groupby(grp_keys).transform(running_adi)
    return df


def run(panel: pd.DataFrame, sales: pd.DataFrame, product: pd.DataFrame) -> dict:
    """Entry point.

    Returns
    -------
    dict with keys:
        classification : per-SKU SBA table (ADI, CV2, Class, Branch)
        panel          : input panel + purchase-frequency feature columns
    """
    classification = classify(sales, product)
    panel_freq = add_frequency_features(panel)
    return {"classification": classification, "panel": panel_freq}


if __name__ == "__main__":
    from . import data_agent
    d = data_agent.run(persist=False)
    res = run(d["panel"], d["tables"]["sales"], d["tables"]["product"])
    print(res["classification"]["Class"].value_counts())
    print("\nNew frequency columns:",
          [c for c in res["panel"].columns if c.startswith("freq") or c == "running_adi"])
