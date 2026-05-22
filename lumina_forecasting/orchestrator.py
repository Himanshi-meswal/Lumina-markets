"""Orchestrator — composes the forecasting tree.

Flow (matches the architecture diagram):

    data_agent
        -> demand_pattern_agent           (classify + freq features)
        -> feature_engineering            (shared backbone)
        -> route by archetype / halo / cold-start:
               standard_agent  (trains the ONE global model, reused below)
               halo_agent      (restricts to halo SKUs, reuses global model)
               npd_agent       (cold-start demo, reuses global model)
        -> reconciliation_agent           (single MinT join point)
        -> inventory_agent                (newsvendor order points)
        -> validation_agent               (deterministic gate)
        -> summarizer_agent               (human rollup)

The global model is trained ONCE in the standard branch and handed to the halo
and NPD branches, so we pay training cost once while keeping branch outputs
cleanly separated (pattern sharing without redundant fits).
"""
from __future__ import annotations
import pandas as pd

from . import data_io
from . import config
from .nodes import (
    data_agent, demand_pattern_agent, feature_engineering,
    standard_agent, halo_agent, npd_agent,
    reconciliation_agent, inventory_agent, validation_agent, summarizer_agent,
)


def route(classification: pd.DataFrame) -> dict[str, list[str]]:
    """Decide which branch each SKU belongs to.

    Priority: halo membership overrides demand-class default. (Cold start is a
    separate trigger handled per new-SKU request, not from the class table.)
    """
    halo = set(halo_agent.halo_skus())
    routing = {"standard": [], "halo": []}
    for _, row in classification.iterrows():
        if row.SKU_ID in halo:
            routing["halo"].append(row.SKU_ID)
        else:
            routing[row.Branch].append(row.SKU_ID)
    return routing


def run(excel_path: str | None = None,
        new_sku_attrs: dict | None = None,
        persist: bool = True,
        llm_client=None) -> dict:
    """Execute the full tree. Returns a state dict with every node's output.

    llm_client : optional callable(prompt)->str passed to the summarizer for a
                 natural-language narrative (used for testing/mocking; the real
                 Gemini call runs automatically when config.LLM_ENABLED is True).
    """
    state: dict = {}

    # 1. data
    d = data_agent.run(excel_path, persist=persist)
    state["panel"] = d["panel"]
    state["tables"] = d["tables"]
    state["data_report"] = d["report"]

    # 2. demand pattern (classification + frequency features)
    dp = demand_pattern_agent.run(d["panel"], d["tables"]["sales"], d["tables"]["product"])
    state["classification"] = dp["classification"]

    # 3. shared features
    feats = feature_engineering.run(dp["panel"])
    state["features"] = feats
    feat_cols = feature_engineering.feature_columns(feats)

    # 4. routing
    routing = route(dp["classification"])
    state["routing"] = {k: len(v) for k, v in routing.items()}

    # 5. train the ONE global model in the standard branch
    train, test, cutoff = standard_agent.train_test_split(feats)
    models = standard_agent.train_global(train, feat_cols)
    state["cutoff"] = cutoff

    # standard predictions (all non-halo SKUs)
    std_preds = standard_agent.predict(models, test[test.SKU_ID.isin(routing["standard"])], feat_cols)
    state["standard"] = {"predictions": std_preds,
                         "importance": standard_agent.feature_importance(models, feat_cols)}

    # halo branch (reuse global model): brand-escalation + two-stage anchor ML
    halo_res = halo_agent.run(feats, d["tables"]["sales"], d["tables"]["pricing"], models=models)
    state["halo"] = halo_res

    # combine branch predictions for downstream nodes. The halo agent may add a
    # 'source' column (direct/brand/anchor_ml) — keep only shared forecast cols.
    keep = ["Date", "Store_ID", "SKU_ID", "Units_Sold", "P10", "P50", "P90"]
    halo_preds = halo_res["predictions"]
    halo_preds = halo_preds[[c for c in keep if c in halo_preds.columns]]
    all_preds = pd.concat([std_preds[keep], halo_preds], ignore_index=True)
    # attach dims needed by reconciliation/inventory
    dims = feats[["Date", "Store_ID", "SKU_ID", "Zone", "Sub_Category", "Perishability_Flag"]].drop_duplicates()
    all_preds = all_preds.merge(dims, on=["Date", "Store_ID", "SKU_ID"], how="left")
    state["predictions"] = all_preds

    # 6. cold-start demo (optional)
    if new_sku_attrs is not None:
        state["npd"] = npd_agent.run(feats, d["tables"]["sales"], d["tables"]["product"],
                                     new_sku_attrs, models=models)

    # 7. reconciliation (single join point)
    state["reconciliation"] = reconciliation_agent.run(
        all_preds,
        method=config.RECON_METHOD,
        zone_capacity_factor=(config.RECON_ZONE_CAPACITY_FACTOR
                              if config.RECON_METHOD == "l1_lp" else None),
        shelf_capacity_factor=(config.RECON_SHELF_CAPACITY_FACTOR
                               if config.RECON_METHOD == "l1_lp" else None),
        moq=config.RECON_MOQ,
        batch=config.RECON_BATCH,
    )

    # 8. inventory
    state["inventory"] = inventory_agent.run(all_preds)

    # 9. validation gate
    state["validation"] = validation_agent.run(all_preds)

    # 10. summarize
    state["summary"] = summarizer_agent.run(state, llm_client=llm_client)

    if persist:
        data_io.save_artifact(all_preds, "predictions_combined")
        data_io.save_artifact(state["summary"], "run_summary")
    return state
