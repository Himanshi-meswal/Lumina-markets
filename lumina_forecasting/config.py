"""Central configuration for the Lumina forecasting tree.

All tunable constants live here so individual nodes stay free of magic numbers.
"""
from __future__ import annotations
import os
from dataclasses import dataclass, field

# ----------------------------------------------------------------------------
# Data location
# ----------------------------------------------------------------------------
# Overridable via environment variables so the SAME container image works for
# any bucket/path without rebuilding. Locally these default to on-disk paths;
# in Cloud Run you set LUMINA_EXCEL_PATH=gs://bucket/file.xlsx etc.
EXCEL_PATH = os.environ.get("LUMINA_EXCEL_PATH", "Lumina_Markets_Dataset.xlsx")
ARTIFACT_DIR = os.environ.get("LUMINA_ARTIFACT_DIR", "artifacts")

SHEETS = {
    "sales": "Fact_Sales",
    "product": "Dim_Product",
    "store": "Dim_Store",
    "pricing": "Fact_Pricing",
    "calendar": "Dim_Calendar",
}

# ----------------------------------------------------------------------------
# Demand classification (Syntetos-Boylan)
# ----------------------------------------------------------------------------
ADI_CUTOFF = 1.32      # avg demand interval threshold
CV2_CUTOFF = 0.49      # squared CoV of non-zero demand sizes threshold

# Map each demand class to the forecasting branch that should handle it.
CLASS_TO_BRANCH = {
    "Smooth":       "standard",
    "Erratic":      "standard",
    "Intermittent": "standard",   # global model w/ quantile loss handles sparsity
    "Lumpy":        "standard",
    "Dead":         "standard",
}
# SKUs flagged as halo (substitution/complement pairs) override the class route.
# NPD route is selected when a SKU has < COLD_START_MIN_DAYS of history.
COLD_START_MIN_DAYS = 28

# ----------------------------------------------------------------------------
# Feature engineering
# ----------------------------------------------------------------------------
LAGS = [1, 7, 14, 28]
ROLL_WINDOWS = [7, 28]
FREQ_WINDOWS = [28, 90]            # trailing windows for purchase-frequency features

CATEGORICAL_COLS = [
    "Category", "Sub_Category", "Brand_Tier", "Zone",
    "Store_Format", "Climate_Zone", "SKU_ID", "Store_ID",
]

# Columns never fed to the model (targets, identifiers, raw aggregates)
NON_FEATURE_COLS = [
    "Date", "Units_Sold", "Revenue", "End_of_Day_Inventory",
    "Promotion_Type", "Base_Price", "Actual_Selling_Price",
    "sub_promo_share", "sub_n", "sub_units",
]

# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
QUANTILES = [0.1, 0.5, 0.9]
TEST_HORIZON_DAYS = 56            # final N days held out for evaluation


@dataclass
class LGBMParams:
    objective: str = "quantile"
    n_estimators: int = 400
    learning_rate: float = 0.05
    num_leaves: int = 63
    min_child_samples: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    verbose: int = -1

    def as_dict(self) -> dict:
        return self.__dict__.copy()


LGBM = LGBMParams()

# ----------------------------------------------------------------------------
# Hierarchy for reconciliation (top -> bottom)
# ----------------------------------------------------------------------------
HIERARCHY = ["TOTAL", "Zone", "SubCat_Zone", "SKU_Store"]

# Reconciliation method: "mint" (unconstrained least squares) or
# "l1_lp" (constrained linear program: non-neg + DC cap + shelf + batch).
RECON_METHOD = "mint"
# Constraint knobs used only when RECON_METHOD == "l1_lp". Factors are fractions
# of the naive bottom-up aggregate; set to None to leave that constraint off.
# Numbers are ILLUSTRATIVE on the synthetic data.
RECON_ZONE_CAPACITY_FACTOR = 0.95     # cap each Zone at 95% of its bottom-up sum
RECON_SHELF_CAPACITY_FACTOR = 1.10    # perishable cells capped at 110% of forecast
RECON_MOQ = 0.0                       # supplier minimum order quantity (post-LP)
RECON_BATCH = 1.0                     # supplier batch multiple (post-LP)

# ----------------------------------------------------------------------------
# Inventory (newsvendor) cost scenarios. Cu = underage (stockout), Co = overage.
# ----------------------------------------------------------------------------
COST_SCENARIOS = {
    "staple":      {"Cu": 4.0, "Co": 0.2},
    "premium_dry": {"Cu": 6.0, "Co": 0.5},
    "perishable":  {"Cu": 3.0, "Co": 2.5},
}

# ----------------------------------------------------------------------------
# Validation tolerances (deterministic gate)
# ----------------------------------------------------------------------------
@dataclass
class Tolerances:
    max_negative_frac: float = 0.0          # forecasts must be >= 0
    min_interval_coverage: float = 0.70     # P10-P90 should cover >= 70%
    max_interval_coverage: float = 0.95
    max_wmape: float = 0.60                 # sanity ceiling on point error
    quantile_monotonic: bool = True         # P10 <= P50 <= P90

TOL = Tolerances()

# ----------------------------------------------------------------------------
# LLM summariser (optional layer on top of the deterministic summary)
# ----------------------------------------------------------------------------
# The deterministic summary always runs. If LLM_ENABLED is True AND an API key
# is present, the summarizer ALSO produces a natural-language brief via Gemini.
# Falls back silently to the deterministic text if the key/SDK is missing.
LLM_ENABLED = False
LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.0-flash"          # fast + cheap; swap for gemini-2.5-pro if needed
LLM_API_KEY_ENV = "GEMINI_API_KEY"      # read the key from this environment variable
LLM_TEMPERATURE = 0.3                   # low — we want faithful reporting, not creativity
LLM_MAX_TOKENS = 600
# Audience preset shapes the prompt: "executive", "analyst", or "planner".
LLM_AUDIENCE = "executive"

# --- Backend selection ------------------------------------------------------
# "aistudio" : standalone API key from aistudio.google.com (simplest).
# "vertex"   : Gemini via your GCP project (uses the $300 free-trial credit,
#              IAM auth, no API key). Set project + location below.
LLM_BACKEND = "aistudio"
# Vertex/GCP settings (only used when LLM_BACKEND == "vertex").
GCP_PROJECT_ENV = "GOOGLE_CLOUD_PROJECT"   # env var holding your GCP project id
GCP_LOCATION = "us-central1"               # a region where Gemini is available
