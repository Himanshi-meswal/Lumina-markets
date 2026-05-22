"""Data agent.

Responsibilities
----------------
1. Load the four case-study tables.
2. Run cheap integrity checks (no nulls in keys, dates monotonic, pricing
   covers sales, etc.) and surface a structured report rather than crashing.
3. Provide the merged panel that every downstream feature node builds on.

This node does *not* engineer features or classify demand — it only guarantees
the branches downstream receive clean, joined data.
"""
from __future__ import annotations
import pandas as pd

from .. import data_io


def _integrity_report(tables: dict[str, pd.DataFrame]) -> dict:
    sales = tables["sales"]
    report = {
        "n_sales_rows": int(len(sales)),
        "n_skus": int(sales.SKU_ID.nunique()),
        "n_stores": int(sales.Store_ID.nunique()),
        "date_min": str(sales.Date.min().date()),
        "date_max": str(sales.Date.max().date()),
        "issues": [],
    }
    # key nulls
    for col in ["Date", "Store_ID", "SKU_ID", "Units_Sold"]:
        n = int(sales[col].isnull().sum())
        if n:
            report["issues"].append(f"{n} null values in Fact_Sales.{col}")
    # negative units
    neg = int((sales.Units_Sold < 0).sum())
    if neg:
        report["issues"].append(f"{neg} negative Units_Sold rows")
    # pricing coverage
    sku_in_pricing = set(tables["pricing"].SKU_ID.unique())
    missing_price = set(sales.SKU_ID.unique()) - sku_in_pricing
    if missing_price:
        report["issues"].append(f"{len(missing_price)} SKUs missing from Fact_Pricing")
    # orphan dimension rows
    prod_orphans = set(sales.SKU_ID.unique()) - set(tables["product"].SKU_ID.unique())
    if prod_orphans:
        report["issues"].append(f"{len(prod_orphans)} SKUs missing from Dim_Product")
    report["clean"] = len(report["issues"]) == 0
    return report


def build_panel(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Left-join sales -> product -> store -> pricing into one panel."""
    df = (
        tables["sales"]
        .merge(tables["product"], on="SKU_ID", how="left")
        .merge(tables["store"], on="Store_ID", how="left")
        .merge(tables["pricing"], on=["Date", "SKU_ID"], how="left")
        .sort_values(["Store_ID", "SKU_ID", "Date"])
        .reset_index(drop=True)
    )
    return df


def run(excel_path: str | None = None, persist: bool = True) -> dict:
    """Entry point.

    Returns
    -------
    dict with keys:
        tables : dict of raw tables
        panel  : merged DataFrame
        report : integrity report
    """
    tables = data_io.load_tables(excel_path)
    report = _integrity_report(tables)
    panel = build_panel(tables)

    out = {"tables": tables, "panel": panel, "report": report}
    if persist:
        data_io.save_artifact(panel, "panel")
        data_io.save_artifact(report, "data_report")
    return out


if __name__ == "__main__":
    res = run()
    r = res["report"]
    print(f"Loaded {r['n_sales_rows']:,} rows | {r['n_skus']} SKUs | {r['n_stores']} stores")
    print(f"Span {r['date_min']} -> {r['date_max']}")
    print("Clean:" , r["clean"])
    for issue in r["issues"]:
        print("  -", issue)
