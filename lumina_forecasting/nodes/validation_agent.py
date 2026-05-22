"""Validation agent — deterministic quality gate.

NOT an LLM agent: a set of cheap, deterministic checks run after forecasting and
before the forecasts are trusted downstream. It works hand-in-hand with the
reconciliation node (coherence) and reports pass/fail against the tolerances in
config.TOL so the orchestrator can decide whether to ship, retry, or fall back.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config


def pinball(y: np.ndarray, yhat: np.ndarray, q: float) -> float:
    d = y - yhat
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def run(predictions: pd.DataFrame, tol=None) -> dict:
    """Entry point. Returns {passed: bool, checks: [...], metrics: {...}}."""
    tol = tol or config.TOL
    qcols = [c for c in predictions.columns if c.startswith("P") and c[1:].isdigit()]
    qcols = sorted(qcols, key=lambda c: int(c[1:]))
    y = predictions["Units_Sold"].to_numpy()
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # 1. non-negativity
    neg_frac = float((predictions[qcols] < 0).to_numpy().mean())
    add("non_negative", neg_frac <= tol.max_negative_frac, f"neg_frac={neg_frac:.4f}")

    # 2. quantile monotonicity P10<=P50<=P90
    if tol.quantile_monotonic and len(qcols) >= 2:
        arr = predictions[qcols].to_numpy()
        mono = bool(np.all(np.diff(arr, axis=1) >= -1e-9))
        add("quantile_monotonic", mono, "non-crossing across quantiles")

    metrics = {}
    # 3. interval coverage (needs lowest & highest quantile)
    if len(qcols) >= 2:
        lo, hi = qcols[0], qcols[-1]
        cov = float(np.mean((y >= predictions[lo].to_numpy()) & (y <= predictions[hi].to_numpy())))
        metrics["interval_coverage"] = round(cov, 3)
        add("interval_coverage",
            tol.min_interval_coverage <= cov <= tol.max_interval_coverage,
            f"{lo}-{hi} coverage={cov:.3f}")

    # 4. point error sanity (WMAPE on median if present)
    if "P50" in qcols:
        denom = np.abs(y).sum()
        wmape = float(np.abs(y - predictions["P50"].to_numpy()).sum() / denom) if denom else 0.0
        metrics["wmape"] = round(wmape, 3)
        add("wmape_ceiling", wmape <= tol.max_wmape, f"wmape={wmape:.3f}")
        for q in qcols:
            metrics[f"pinball_{q}"] = round(pinball(y, predictions[q].to_numpy(), int(q[1:]) / 100), 3)

    passed = all(c["pass"] for c in checks)
    return {"passed": passed, "checks": checks, "metrics": metrics}


if __name__ == "__main__":
    from .. import data_io
    preds = data_io.load_artifact("stage3_predictions") if data_io.artifact_exists("stage3_predictions") else None
    if preds is None:
        print("Run the forecasting branches first.")
    else:
        res = run(preds)
        print("PASSED" if res["passed"] else "FAILED")
        for c in res["checks"]:
            print(f"  [{'ok' if c['pass'] else 'XX'}] {c['check']}: {c['detail']}")
        print("metrics:", res["metrics"])
