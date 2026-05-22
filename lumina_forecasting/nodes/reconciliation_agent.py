"""Reconciliation agent — coherence join point (4.4).

Two reconciliation methods are available:

  * `mint_reconcile`        : classic MinT (unconstrained least squares). Fast,
                              statistically optimal, but can return negative or
                              operationally-infeasible quantities.
  * `reconcile_l1_lp`       : L1 (least-absolute-deviation) reconciliation cast
                              as a LINEAR PROGRAM, with real-world constraints:
                              non-negativity, DC/warehouse capacity, perishable
                              shelf-space. More robust to outliers and always
                              feasible. Supplier MOQ / batch sizing is applied
                              as a clearly-labelled post-LP rounding step (that
                              part is integer logic, not LP).

This is a SINGLE join point that runs once, AFTER every branch has produced its
base forecasts.

LP formulation
--------------
Decision vars: bottom quantities b (len n_bottom) and aux deviations u (len n_nodes).
Reconciled full vector  y_tilde = S b  (coherence is automatic — upper nodes are
sums of the bottom decision variables).

    minimise   sum_i  w_i * u_i
    s.t.       (S b)_i - yhat_i <= u_i           for all nodes i   (|.| upper leg)
              -(S b)_i + yhat_i <= u_i           for all nodes i   (|.| lower leg)
               b >= 0                                              (non-negativity)
               (S b)_z <= cap_z                  zone rows         (DC capacity)
               b_j <= shelf_j                    perishable cells  (shelf space)

The two |.| legs are the standard linearisation of the absolute value.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .. import data_io


# ----------------------------------------------------------------------------
# Summing matrix (shared by both methods)
# ----------------------------------------------------------------------------
def build_summing_matrix(bottom: pd.DataFrame):
    """Construct S mapping bottom series -> [upper nodes; bottom identity].

    `bottom` must have columns: bottom_id, Zone, Sub_Category.
    Returns (S, upper_node_names, bottom_df_with_scz).
    """
    bottom = bottom.reset_index(drop=True)
    bottom["scz"] = bottom.Sub_Category.astype(str) + "@" + bottom.Zone.astype(str)
    scz_nodes = bottom.scz.unique().tolist()
    zone_nodes = bottom.Zone.astype(str).unique().tolist()
    upper = scz_nodes + zone_nodes + ["TOTAL"]

    n_bottom = len(bottom)
    n_upper = len(upper)
    S = np.zeros((n_upper + n_bottom, n_bottom))
    for r, node in enumerate(upper):
        if node == "TOTAL":
            S[r, :] = 1.0
        elif node in set(zone_nodes):
            S[r, (bottom.Zone.astype(str) == node).to_numpy()] = 1.0
        else:
            S[r, (bottom.scz == node).to_numpy()] = 1.0
    S[n_upper:, :] = np.eye(n_bottom)
    return S, upper, bottom


# ----------------------------------------------------------------------------
# Method 1: classic MinT (unconstrained)
# ----------------------------------------------------------------------------
def mint_reconcile(S: np.ndarray, base_all: np.ndarray, weight: np.ndarray | None = None):
    """Return reconciled bottom-level vector via MinT (least squares)."""
    if weight is None:
        G = np.linalg.inv(S.T @ S) @ S.T               # OLS
    else:
        Wi = np.linalg.inv(weight)
        G = np.linalg.inv(S.T @ Wi @ S) @ S.T @ Wi     # GLS / MinT
    return G @ base_all


# ----------------------------------------------------------------------------
# Method 2: L1 reconciliation as a Linear Program (operational constraints)
# ----------------------------------------------------------------------------
def reconcile_l1_lp(S: np.ndarray,
                    base_all: np.ndarray,
                    upper: list,
                    bottom: pd.DataFrame,
                    weights: np.ndarray | None = None,
                    zone_capacity: dict | None = None,
                    shelf_capacity: dict | None = None,
                    perishable_mask: np.ndarray | None = None) -> dict:
    """Solve the L1 reconciliation LP. Returns reconciled bottom vector + diag."""
    n_nodes, n_bottom = S.shape
    if weights is None:
        weights = np.ones(n_nodes)

    # Decision vector x = [ b (n_bottom) ; u (n_nodes) ]
    # Objective: minimise sum w_i u_i  (b has zero cost)
    c = np.concatenate([np.zeros(n_bottom), weights])

    # Absolute-value linearisation:
    #   S b - u <= yhat     ->  [ S , -I ] x <= base_all
    #  -S b - u <= -yhat    ->  [-S , -I ] x <= -base_all
    A_ub = np.vstack([
        np.hstack([S, -np.eye(n_nodes)]),
        np.hstack([-S, -np.eye(n_nodes)]),
    ])
    b_ub = np.concatenate([base_all, -base_all])

    # DC / warehouse capacity: (S b)_zone <= cap
    if zone_capacity:
        zone_rows, zone_caps = [], []
        for r, node in enumerate(upper):
            if node in zone_capacity:
                zone_rows.append(np.concatenate([S[r], np.zeros(n_nodes)]))
                zone_caps.append(zone_capacity[node])
        if zone_rows:
            A_ub = np.vstack([A_ub, np.array(zone_rows)])
            b_ub = np.concatenate([b_ub, np.array(zone_caps)])

    # Bounds: b >= 0 (and <= shelf cap for perishable cells); u >= 0
    b_upper_bounds = [None] * n_bottom
    if shelf_capacity and perishable_mask is not None:
        for j in range(n_bottom):
            bid = bottom.iloc[j]["bottom_id"]
            if perishable_mask[j] and bid in shelf_capacity:
                b_upper_bounds[j] = shelf_capacity[bid]
    bounds = [(0, b_upper_bounds[j]) for j in range(n_bottom)] + [(0, None)] * n_nodes

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(f"LP did not solve: {res.message}")
    return {"recon_bottom": res.x[:n_bottom], "status": res.message, "objective": float(res.fun)}


def apply_moq_batch(recon_bottom: np.ndarray, moq: float = 0.0, batch: float = 1.0) -> np.ndarray:
    """Post-LP supplier rounding (INTEGER logic — not part of the LP).

    Round UP to nearest batch multiple, then enforce MOQ (snap small orders to
    0 or up to MOQ, nearest wins). A fully optimal MOQ solution needs a MILP,
    which we deliberately avoid here for speed/clarity.
    """
    out = recon_bottom.copy().astype(float)
    if batch and batch > 1:
        out = np.ceil(out / batch) * batch
    if moq and moq > 0:
        small = (out > 0) & (out < moq)
        out[small] = np.where(out[small] >= moq / 2.0, moq, 0.0)
    return out


# ----------------------------------------------------------------------------
# Orchestrated entry point
# ----------------------------------------------------------------------------
def run(predictions: pd.DataFrame,
        quantile_col: str = "P50",
        day=None,
        upper_bias: float = 0.12,
        method: str = "mint",
        zone_capacity_factor: float | None = None,
        shelf_capacity_factor: float | None = None,
        moq: float = 0.0,
        batch: float = 1.0) -> dict:
    """Entry point. method = 'mint' (unconstrained) or 'l1_lp' (constrained LP)."""
    day = day or predictions.Date.max()
    d = predictions[predictions.Date == day].copy()
    d["bottom_id"] = d.SKU_ID.astype(str) + "|" + d.Store_ID.astype(str)

    agg = {"yhat": (quantile_col, "sum"),
           "Zone": ("Zone", "first"),
           "Sub_Category": ("Sub_Category", "first")}
    if "Perishability_Flag" in d.columns:
        agg["Perishability_Flag"] = ("Perishability_Flag", "max")
    bottom = d.groupby("bottom_id").agg(**agg).reset_index()

    S, upper, bottom = build_summing_matrix(bottom)
    yb = bottom.yhat.to_numpy()
    n_upper = len(upper)

    rng = np.random.default_rng(0)
    base_upper = (S[:n_upper] @ yb) * (1 + upper_bias) * (1 + 0.05 * rng.standard_normal(n_upper))
    base_all = np.concatenate([base_upper, yb])
    i_total = upper.index("TOTAL")

    extra = {}
    mint_for_compare = mint_reconcile(S, base_all)

    if method == "mint":
        recon_bottom = mint_for_compare

    elif method == "l1_lp":
        zone_capacity = None
        if zone_capacity_factor is not None:
            zone_capacity = {}
            zoneset = set(bottom.Zone.astype(str))
            for r, node in enumerate(upper):
                if node in zoneset:
                    zone_capacity[node] = float(S[r] @ yb) * zone_capacity_factor

        shelf_capacity = None
        perishable_mask = None
        if "Perishability_Flag" in bottom.columns:
            perishable_mask = (bottom.Perishability_Flag.to_numpy() == 1)
            if shelf_capacity_factor is not None:
                shelf_capacity = {bottom.iloc[j]["bottom_id"]: float(yb[j]) * shelf_capacity_factor
                                  for j in range(len(bottom)) if perishable_mask[j]}

        lp = reconcile_l1_lp(S, base_all, upper, bottom,
                             zone_capacity=zone_capacity,
                             shelf_capacity=shelf_capacity,
                             perishable_mask=perishable_mask)
        recon_bottom = lp["recon_bottom"]
        extra["lp_status"] = lp["status"]
        extra["lp_objective"] = round(lp["objective"], 2)

        if moq or (batch and batch > 1):
            rounded = apply_moq_batch(recon_bottom, moq=moq, batch=batch)
            extra["moq_batch"] = {"moq": moq, "batch": batch}
            extra["units_added_by_rounding"] = round(float((rounded - recon_bottom).clip(min=0).sum()), 1)
            recon_bottom = rounded
    else:
        raise ValueError(f"Unknown method '{method}' (use 'mint' or 'l1_lp')")

    recon_all = S @ recon_bottom
    report = {
        "method": method,
        "day": str(pd.Timestamp(day).date()),
        "n_bottom": int(len(bottom)),
        "n_upper": int(n_upper),
        "incoherent_total": round(float(base_all[i_total]), 1),
        "reconciled_total": round(float(recon_all[i_total]), 1),
        "reconciled_bottom_sum": round(float(recon_bottom.sum()), 1),
        "pre_gap_pct": round(float(abs(base_all[i_total] - yb.sum()) / yb.sum() * 100), 1),
        "post_gap": float(abs(recon_all[i_total] - recon_bottom.sum())),
        "n_negative_mint": int((mint_for_compare < -1e-6).sum()),
        "n_negative_after": int((recon_bottom < -1e-6).sum()),
        **extra,
    }
    reconciled = bottom[["bottom_id", "Zone", "Sub_Category"]].copy()
    reconciled["reconciled_qty"] = recon_bottom
    return {"report": report, "reconciled_bottom": reconciled}


if __name__ == "__main__":
    preds = data_io.load_artifact("stage3_predictions") if data_io.artifact_exists("stage3_predictions") else None
    if preds is None:
        print("Run the forecasting branches first to produce predictions.")
    else:
        print("--- MinT (unconstrained) ---")
        for k, v in run(preds, method="mint")["report"].items():
            print(f"  {k}: {v}")
        print("\n--- L1-LP (non-neg + DC cap 0.95 + shelf cap 1.1 + batch 5) ---")
        for k, v in run(preds, method="l1_lp", zone_capacity_factor=0.95,
                        shelf_capacity_factor=1.1, batch=5)["report"].items():
            print(f"  {k}: {v}")
