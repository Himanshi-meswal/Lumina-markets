# Lumina Forecasting Engine — modular node tree

A demand-forecasting system for the Lumina Markets case study, composed as a tree
of small, independently testable `.py` nodes rather than one monolithic script.

## Layout

```
lumina_forecasting/
├── config.py                 # all constants, hyperparameters, tolerances
├── data_io.py                # table loading + artifact persistence
├── orchestrator.py           # composes the tree, routes SKUs by archetype
├── run_pipeline.py           # CLI entrypoint
└── nodes/
    ├── data_agent.py             # load · validate · merge panel
    ├── demand_pattern_agent.py   # Syntetos-Boylan class + purchase-freq features
    ├── feature_engineering.py    # calendar · lags · price · cross-product
    ├── standard_agent.py         # global LightGBM quantile model (core forecaster)
    ├── halo_agent.py             # substitution + complementarity SKUs
    ├── npd_agent.py              # cold-start analog borrowing
    ├── reconciliation_agent.py   # MinT coherence join point
    ├── inventory_agent.py        # newsvendor order points from quantiles
    ├── validation_agent.py       # deterministic quality gate
    └── summarizer_agent.py       # human-readable rollup
```

## Node contract

Every node exposes a single `run(...)` function that takes plain objects
(DataFrames / dicts) and returns plain objects. No node touches disk except via
`data_io`. This makes each node:
- runnable standalone:  `python -m lumina_forecasting.nodes.<node>`
- unit-testable with a synthetic DataFrame
- swappable without touching its neighbours

## Run the whole tree

```bash
# point config.EXCEL_PATH at your workbook, or pass --excel
python -m lumina_forecasting.run_pipeline --excel Lumina_Markets_Dataset.xlsx --cold-start
```

## Tree flow

```
data_agent
  └─ demand_pattern_agent      (classify + leakage-safe frequency features)
       └─ feature_engineering  (shared backbone)
            └─ orchestrator routes by archetype:
                 standard_agent   ← trains the ONE global model, shared below
                 halo_agent       ← reuses model, restricts to halo SKUs
                 npd_agent        ← reuses model, cold-start analogs
            └─ reconciliation_agent  (single MinT join point, after all branches)
            └─ inventory_agent       (newsvendor)
            └─ validation_agent      (deterministic gate)
            └─ summarizer_agent      (rollup)
```

## Key design decisions

- **One global model, trained once.** Trained in the standard branch and handed
  to halo/NPD, so sparse SKUs share patterns with dense ones and we pay the fit
  cost once.
- **Classification precedes routing.** The orchestrator routes on the SBA class,
  so demand-pattern characterization is a *pre-step*, not a downstream agent.
- **Reconciliation is a single join point**, after every branch finishes.
- **Validation is deterministic**, not an LLM agent — cheap pass/fail checks.
- **Frequency features are leakage-safe** — all trailing windows use `shift(1)`
  before rolling so a row never sees its own day's sale.

## Notes

- Magnitudes are illustrative (synthetic data, seed 42), not calibrated to a real
  retailer.
- The intermittent branch currently uses the global quantile model. For tighter
  P50 calibration on 90%-zero series, swap in a Tweedie objective or a two-stage
  hurdle model inside `standard_agent.train_global`.
```
