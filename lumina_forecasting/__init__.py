"""Lumina Markets demand-forecasting engine — modular agent/node package.

Tree composition:

    run_pipeline.py
        └── orchestrator.run()
                ├── data_agent.run()                 (load + validate + route)
                ├── demand_pattern_agent.run()        (SBA class + freq features)
                ├── feature_engineering.run()         (shared feature backbone)
                ├── {standard|halo|npd}_agent.run()   (branch forecasters)
                ├── reconciliation_agent.run()        (MinT coherence join point)
                ├── inventory_agent.run()             (newsvendor order points)
                ├── validation_agent.run()            (deterministic gate)
                └── summarizer_agent.run()            (human rollup)

Every node exposes a `run(...)` function that takes and returns plain objects
(DataFrames / dicts), so nodes can be composed, tested, or swapped in isolation.
"""
__version__ = "0.1.0"
