"""Halo agent — brand-level forecasting + ML anchor modelling (4.2).

This agent handles the two cross-product dynamics from the case study with
dedicated ML, not just engineered features:

A. SUBSTITUTION via brand-level escalation
   Individual halo SKUs are noisy: cannibalisation between shelf-neighbours
   swamps the own-SKU signal. So for each halo SKU we ALSO forecast at the
   BRAND level (Category x Brand_Tier), where sibling cannibalisation nets out
   and the series is denser / more stable. We then:
     1. train an ML quantile model on brand-level aggregated demand,
     2. disaggregate the brand forecast back to SKUs by each SKU's historical
        share of its brand,
     3. compare the brand-routed forecast against the direct per-SKU forecast
        on the holdout, and KEEP whichever has higher confidence per SKU.
   "Confidence" blends interval tightness and holdout error (see _confidence).
   This directly answers: did raising to brand level actually help this SKU?

B. COMPLEMENTARITY via a two-stage ML anchor model
   Anchor (cereal) -> attach (milk). Rather than copy a lag, we:
     1. forecast the anchor's demand with the global model,
     2. feed that PREDICTED anchor demand as a feature into a dedicated
        attach model (LightGBM quantile) that learns a nonlinear response.
   The attach forecast therefore reacts to *expected* anchor demand, which is
   what genuinely drives complementary purchases.

Membership (which SKUs are halo, and the anchor/attach pairs) is a business
input, kept explicit so the feature/aggregation set stays bounded.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import lightgbm as lgb

from .. import config
from ..nodes import standard_agent, feature_engineering


# ----------------------------------------------------------------------------
# Cross-product groups (business input). In production these come from
# merchandising config or market-basket lift mining.
# ----------------------------------------------------------------------------
HALO_PAIRS = {
    "SODA":  {"private": "SKU0010", "premium": "SKU0011"},
    "CHIP":  {"private": "SKU0012", "premium": "SKU0013"},
    "BFAST": {"anchor": "SKU0020", "attach": "SKU0021"},   # cereal -> milk
    "BBQ":   {"anchor": "SKU0022", "attach": "SKU0023"},   # buns  -> beef
}

# Brand-level grouping key. No standalone Brand column exists in the dataset, so
# brand := (Category, Brand_Tier). Swap for a real 'Brand' column in production.
BRAND_KEYS = ["Category", "Brand_Tier"]


def halo_skus() -> list[str]:
    out = []
    for grp in HALO_PAIRS.values():
        out.extend(grp.values())
    return sorted(set(out))


def anchor_pairs() -> dict:
    """Subset of HALO_PAIRS that are anchor->attach (complementarity)."""
    return {k: v for k, v in HALO_PAIRS.items() if "anchor" in v and "attach" in v}


# ----------------------------------------------------------------------------
# Diagnostic: measured substitution effect (kept from the original agent)
# ----------------------------------------------------------------------------
def measure_substitution(sales: pd.DataFrame, pricing: pd.DataFrame, pair: dict):
    if "private" not in pair or "premium" not in pair:
        return None
    priv = pricing[pricing.SKU_ID == pair["private"]][["Date", "Promotion_Type"]].copy()
    priv["promo"] = (priv.Promotion_Type != "No_Promo").astype(int)
    prem_u = sales[sales.SKU_ID == pair["premium"]].groupby("Date").Units_Sold.sum().rename("prem")
    m = priv.merge(prem_u, on="Date")
    base = m.loc[m.promo == 0, "prem"].mean()
    promo = m.loc[m.promo == 1, "prem"].mean()
    return {"premium_baseline": round(float(base), 1),
            "premium_when_rival_promo": round(float(promo), 1),
            "lift_pct": round(float((promo - base) / base * 100), 1)}


# ----------------------------------------------------------------------------
# Confidence scoring: lower is better. Blends interval tightness + holdout error.
# ----------------------------------------------------------------------------
def _confidence(pred_df: pd.DataFrame) -> float:
    """Return a confidence COST for a set of predictions (lower = more confident).

    cost = w1 * normalised_interval_width  +  w2 * wmape
    Both terms are scale-free so brand vs direct are comparable.
    """
    y = pred_df["Units_Sold"].to_numpy()
    p10, p50, p90 = pred_df["P10"].to_numpy(), pred_df["P50"].to_numpy(), pred_df["P90"].to_numpy()
    scale = max(np.mean(np.abs(y)), 1e-6)
    interval = np.mean(p90 - p10) / scale                       # tightness
    denom = np.sum(np.abs(y))
    wmape = np.sum(np.abs(y - p50)) / denom if denom > 0 else np.mean(np.abs(p50 - y))
    return float(0.5 * interval + 0.5 * wmape)


# ----------------------------------------------------------------------------
# A. Brand-level forecasting + disaggregation
# ----------------------------------------------------------------------------
def _build_brand_panel(features: pd.DataFrame) -> pd.DataFrame:
    """Aggregate the feature panel to brand x store x day.

    Sums the target; takes promo/price intensity as means; rebuilds calendar.
    Brand-level lags/rollings are recomputed on the aggregated series.
    """
    f = features.copy()
    f["Brand"] = f[BRAND_KEYS].astype(str).agg(" | ".join, axis=1)

    grp = f.groupby(["Brand", "Store_ID", "Date"], observed=True)
    brand = grp.agg(
        Units_Sold=("Units_Sold", "sum"),
        is_promo=("is_promo", "mean"),
        discount_depth=("discount_depth", "mean"),
        competitor_promo_pressure=("competitor_promo_pressure", "mean"),
        Zone=("Zone", "first"),
        Store_Format=("Store_Format", "first"),
        Climate_Zone=("Climate_Zone", "first"),
    ).reset_index()

    # calendar
    brand["dow"] = brand.Date.dt.dayofweek
    brand["month"] = brand.Date.dt.month
    brand["weekofyear"] = brand.Date.dt.isocalendar().week.astype(int)
    brand["is_weekend"] = (brand.dow >= 5).astype(int)
    doy = brand.Date.dt.dayofyear
    brand["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    brand["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)

    # brand-level lags / rolling
    brand = brand.sort_values(["Brand", "Store_ID", "Date"])
    g = brand.groupby(["Brand", "Store_ID"], observed=True)["Units_Sold"]
    for lag in config.LAGS:
        brand[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(1)
    keys = [brand.Brand, brand.Store_ID]
    for win in config.ROLL_WINDOWS:
        brand[f"roll_mean_{win}"] = shifted.groupby(keys).transform(
            lambda s: s.rolling(win, min_periods=1).mean())
        brand[f"roll_std_{win}"] = shifted.groupby(keys).transform(
            lambda s: s.rolling(win, min_periods=1).std())

    for c in ["Brand", "Zone", "Store_Format", "Climate_Zone", "Store_ID"]:
        brand[c] = brand[c].astype("category")
    return brand.reset_index(drop=True)


def _brand_feature_cols(brand_panel: pd.DataFrame) -> list[str]:
    drop = {"Date", "Units_Sold"}
    return [c for c in brand_panel.columns if c not in drop]


def _historical_share(features: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Each SKU's share of its brand's units, by store, computed on TRAIN only."""
    f = features[features.Date <= cutoff].copy()
    f["Brand"] = f[BRAND_KEYS].astype(str).agg(" | ".join, axis=1)
    sku_tot = f.groupby(["Brand", "Store_ID", "SKU_ID"], observed=True)["Units_Sold"].sum()
    brand_tot = f.groupby(["Brand", "Store_ID"], observed=True)["Units_Sold"].sum()
    share = (sku_tot / brand_tot).rename("share").reset_index()
    share["share"] = share["share"].fillna(0.0)
    return share


def forecast_brand_level(features: pd.DataFrame,
                         halo: list[str],
                         horizon_days: int | None = None) -> pd.DataFrame:
    """Train a brand-level quantile model, predict the holdout, disaggregate to SKUs."""
    brand_panel = _build_brand_panel(features)
    bcols = _brand_feature_cols(brand_panel)
    bcat = [c for c in ["Brand", "Zone", "Store_Format", "Climate_Zone", "Store_ID"] if c in bcols]

    h = horizon_days or config.TEST_HORIZON_DAYS
    cutoff = brand_panel.Date.max() - pd.Timedelta(days=h)
    btrain = brand_panel[brand_panel.Date <= cutoff].dropna(subset=[f"lag_{max(config.LAGS)}"])
    btest = brand_panel[brand_panel.Date > cutoff]

    # which brands do the halo SKUs belong to?
    fmap = features[["SKU_ID"] + BRAND_KEYS].drop_duplicates()
    fmap["Brand"] = fmap[BRAND_KEYS].astype(str).agg(" | ".join, axis=1)
    halo_brands = set(fmap[fmap.SKU_ID.isin(halo)]["Brand"])

    # train quantile models on brand demand
    models = {}
    X, y = btrain[bcols], btrain["Units_Sold"]
    for q in config.QUANTILES:
        m = lgb.LGBMRegressor(alpha=q, **config.LGBM.as_dict())
        m.fit(X, y, categorical_feature=bcat)
        models[q] = m

    bt = btest[btest.Brand.isin(halo_brands)].copy()
    qs = sorted(models)
    arr = np.column_stack([np.clip(models[q].predict(bt[bcols]), 0, None) for q in qs])
    arr = np.sort(arr, axis=1)
    for i, q in enumerate(qs):
        bt[f"brand_P{int(q*100)}"] = arr[:, i]

    # disaggregate brand forecast to SKUs via historical share
    share = _historical_share(features, cutoff)
    share["Brand"] = share["Brand"].astype(str)
    bt["Brand"] = bt["Brand"].astype(str)
    halo_share = share[share.SKU_ID.isin(halo)]

    merged = halo_share.merge(
        bt[["Brand", "Store_ID", "Date", "brand_P10", "brand_P50", "brand_P90"]],
        on=["Brand", "Store_ID"], how="inner")
    for q in [10, 50, 90]:
        merged[f"P{q}"] = merged[f"brand_P{q}"] * merged["share"]
    return merged[["Date", "Store_ID", "SKU_ID", "P10", "P50", "P90"]]


# ----------------------------------------------------------------------------
# B. Two-stage ML anchor model (complementarity)
# ----------------------------------------------------------------------------
def forecast_anchor_attach(features: pd.DataFrame,
                           global_models: dict,
                           feat_cols: list[str],
                           horizon_days: int | None = None) -> dict:
    """Stage 1: predict anchor demand. Stage 2: attach model uses it as a feature."""
    train, test, cutoff = standard_agent.train_test_split(features, horizon_days)
    pairs = anchor_pairs()
    results = {}

    for name, pair in pairs.items():
        anc, att = pair["anchor"], pair["attach"]

        # --- Stage 1: anchor demand predictions (P50) for ALL rows we need ---
        # predict anchor on the full timeline (train+test) at store-day grain
        anc_rows = features[features.SKU_ID == anc].copy()
        anc_rows["anchor_pred"] = np.clip(global_models[0.5].predict(anc_rows[feat_cols]), 0, None)
        anc_key = anc_rows[["Date", "Store_ID", "anchor_pred"]]

        # --- Stage 2: attach model with anchor_pred as an extra feature ---
        att_all = features[features.SKU_ID == att].merge(anc_key, on=["Date", "Store_ID"], how="left")
        att_all["anchor_pred"] = att_all["anchor_pred"].fillna(0.0)
        # merges can drop the 'category' dtype LightGBM needs; restore it
        for c in config.CATEGORICAL_COLS:
            if c in att_all.columns:
                att_all[c] = att_all[c].astype("category")
        att_cols = feat_cols + ["anchor_pred"]

        att_train = att_all[att_all.Date <= cutoff].dropna(subset=[f"lag_{max(config.LAGS)}"])
        att_test = att_all[att_all.Date > cutoff]
        if len(att_train) < 50 or len(att_test) == 0:
            continue

        cat = [c for c in config.CATEGORICAL_COLS if c in att_cols]
        models = {}
        for q in config.QUANTILES:
            m = lgb.LGBMRegressor(alpha=q, **config.LGBM.as_dict())
            m.fit(att_train[att_cols], att_train["Units_Sold"], categorical_feature=cat)
            models[q] = m

        out = att_test[["Date", "Store_ID", "SKU_ID", "Units_Sold"]].copy()
        qs = sorted(models)
        arr = np.column_stack([np.clip(models[q].predict(att_test[att_cols]), 0, None) for q in qs])
        arr = np.sort(arr, axis=1)
        for i, q in enumerate(qs):
            out[f"P{int(q*100)}"] = arr[:, i]

        # how important was the anchor signal in the attach P50 model?
        imp = pd.Series(models[0.5].feature_importances_, index=att_cols)
        anchor_rank = int(imp.rank(ascending=False)[ "anchor_pred"]) if "anchor_pred" in imp else None

        results[name] = {
            "attach_sku": att, "anchor_sku": anc,
            "predictions": out,
            "anchor_feature_rank": anchor_rank,
            "anchor_feature_importance": int(imp.get("anchor_pred", 0)),
            "n_features": len(att_cols),
        }
    return results


# ----------------------------------------------------------------------------
# Orchestrated entry point
# ----------------------------------------------------------------------------
def run(features: pd.DataFrame,
        sales: pd.DataFrame,
        pricing: pd.DataFrame,
        models: dict | None = None,
        horizon_days: int | None = None) -> dict:
    """Entry point.

    Returns
    -------
    predictions : final per-SKU halo forecast (best of direct vs brand-routed,
                  plus the two-stage anchor/attach forecasts).
    routing_decision : per-SKU which path won and its confidence cost.
    substitution_diagnostics, anchor_diagnostics, cutoff.
    """
    feat_cols = feature_engineering.feature_columns(features)
    halo = halo_skus()
    anchors = anchor_pairs()
    attach_skus = {p["attach"] for p in anchors.values()}

    if models is None:
        train, test, cutoff = standard_agent.train_test_split(features, horizon_days)
        models = standard_agent.train_global(train, feat_cols)
    else:
        _, test, cutoff = standard_agent.train_test_split(features, horizon_days)

    # --- direct per-SKU forecast (baseline to compare brand against) ---
    direct = standard_agent.predict(models, test[test.SKU_ID.isin(halo)], feat_cols)

    # --- brand-level forecast, disaggregated to SKUs ---
    brand = forecast_brand_level(features, halo, horizon_days)
    # attach actuals to brand preds for confidence scoring
    actuals = test[["Date", "Store_ID", "SKU_ID", "Units_Sold"]]
    brand = brand.merge(actuals, on=["Date", "Store_ID", "SKU_ID"], how="inner")

    # --- per-SKU: choose direct vs brand by confidence cost (lower wins) ---
    routing_decision = {}
    chosen_rows = []
    for sku in halo:
        d_sku = direct[direct.SKU_ID == sku]
        b_sku = brand[brand.SKU_ID == sku]
        d_cost = _confidence(d_sku) if len(d_sku) else np.inf
        b_cost = _confidence(b_sku) if len(b_sku) else np.inf
        use_brand = b_cost < d_cost
        routing_decision[sku] = {
            "direct_cost": None if np.isinf(d_cost) else round(d_cost, 4),
            "brand_cost": None if np.isinf(b_cost) else round(b_cost, 4),
            "chosen": "brand" if use_brand else "direct",
            "brand_helped": bool(use_brand),
        }
        chosen_rows.append((b_sku if use_brand else d_sku).assign(source=("brand" if use_brand else "direct")))

    halo_predictions = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else pd.DataFrame()

    # --- two-stage ML anchor/attach (overrides attach SKUs with the richer model) ---
    anchor_res = forecast_anchor_attach(features, models, feat_cols, horizon_days)
    if anchor_res:
        anc_preds = pd.concat([r["predictions"].assign(source="anchor_ml")
                               for r in anchor_res.values()], ignore_index=True)
        # replace attach-sku rows with the two-stage forecast
        halo_predictions = halo_predictions[~halo_predictions.SKU_ID.isin(attach_skus)]
        halo_predictions = pd.concat([halo_predictions, anc_preds], ignore_index=True)

    substitution = {name: measure_substitution(sales, pricing, pair)
                    for name, pair in HALO_PAIRS.items()
                    if measure_substitution(sales, pricing, pair) is not None}
    anchor_diag = {name: {k: r[k] for k in
                          ("anchor_sku", "attach_sku", "anchor_feature_rank",
                           "anchor_feature_importance", "n_features")}
                   for name, r in anchor_res.items()}

    return {"predictions": halo_predictions,
            "halo_skus": halo,
            "routing_decision": routing_decision,
            "substitution_diagnostics": substitution,
            "anchor_diagnostics": anchor_diag,
            "cutoff": cutoff}


if __name__ == "__main__":
    from . import data_agent, demand_pattern_agent
    d = data_agent.run(persist=False)
    dp = demand_pattern_agent.run(d["panel"], d["tables"]["sales"], d["tables"]["product"])
    feats = feature_engineering.run(dp["panel"])
    res = run(feats, d["tables"]["sales"], d["tables"]["pricing"])
    print("=== Routing decision (direct vs brand) ===")
    for sku, info in res["routing_decision"].items():
        print(f"  {sku}: chose {info['chosen']:6s} "
              f"(direct={info['direct_cost']}, brand={info['brand_cost']}, "
              f"brand_helped={info['brand_helped']})")
    print("\n=== Anchor two-stage ML ===")
    for name, info in res["anchor_diagnostics"].items():
        print(f"  {name}: {info['anchor_sku']} -> {info['attach_sku']} | "
              f"anchor_pred rank {info['anchor_feature_rank']}/{info['n_features']}")
    print("\n=== Substitution diagnostics ===")
    for k, v in res["substitution_diagnostics"].items():
        print(f"  {k}: {v}")
