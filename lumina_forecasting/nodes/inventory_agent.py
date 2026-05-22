"""Inventory agent — newsvendor order points from the predictive distribution (4.5).

A single number can't balance empty shelves against wasted food. Given P10/P50/P90
and item economics, the optimal order quantity is the demand quantile at the
critical ratio:

        CR = Cu / (Cu + Co)

  Cu = underage (stockout) cost, Co = overage (spoilage/holding) cost.
High spoilage cost pulls the order point down toward P50/P10; high stockout cost
pushes it up toward P90. Same forecast distribution, item-specific order point.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config


def critical_ratio(Cu: float, Co: float) -> float:
    return Cu / (Cu + Co)


def quantile_from_cr(cr: float) -> str:
    """Map a critical ratio to the nearest available forecast quantile column."""
    avail = config.QUANTILES
    target = min(avail, key=lambda q: abs(q - cr))
    return f"P{int(target * 100)}"


def order_points(predictions: pd.DataFrame,
                 scenario_for_sku=None) -> pd.DataFrame:
    """Attach an order quantity per row based on the item's cost scenario.

    `scenario_for_sku(row) -> scenario_key` chooses economics per SKU. Default
    uses Perishability_Flag: perishable -> 'perishable', else 'staple'.
    """
    if scenario_for_sku is None:
        def scenario_for_sku(row):
            return "perishable" if row.get("Perishability_Flag", 0) == 1 else "staple"

    out = predictions.copy()
    cr_vals, qcols, orders = [], [], []
    for _, row in out.iterrows():
        scen = config.COST_SCENARIOS[scenario_for_sku(row)]
        cr = critical_ratio(scen["Cu"], scen["Co"])
        qcol = quantile_from_cr(cr)
        cr_vals.append(round(cr, 3))
        qcols.append(qcol)
        orders.append(float(np.ceil(row[qcol])))
    out["critical_ratio"] = cr_vals
    out["order_quantile"] = qcols
    out["order_qty"] = orders
    return out


def scenario_table() -> pd.DataFrame:
    """Summary of how each cost scenario maps to an order quantile."""
    rows = []
    for name, c in config.COST_SCENARIOS.items():
        cr = critical_ratio(c["Cu"], c["Co"])
        rows.append({"scenario": name, "Cu": c["Cu"], "Co": c["Co"],
                     "critical_ratio": round(cr, 3), "order_quantile": quantile_from_cr(cr)})
    return pd.DataFrame(rows)


def run(predictions: pd.DataFrame, scenario_for_sku=None) -> dict:
    """Entry point. Returns per-row order points + the scenario summary."""
    detailed = order_points(predictions, scenario_for_sku)
    return {"orders": detailed, "scenarios": scenario_table()}


if __name__ == "__main__":
    from .. import data_io
    preds = data_io.load_artifact("stage3_predictions") if data_io.artifact_exists("stage3_predictions") else None
    if preds is None:
        print("Run the forecasting branches first.")
    else:
        res = run(preds)
        print(res["scenarios"].to_string(index=False))
        print("\nSample orders:")
        print(res["orders"][["SKU_ID", "Perishability_Flag", "P50", "order_quantile", "order_qty"]].head().to_string(index=False))
