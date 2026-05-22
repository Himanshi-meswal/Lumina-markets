"""Summarizer agent — assembles a human-readable rollup of the whole run.

Two layers:
  1. DETERMINISTIC core (build_summary + render_text): turns the pipeline state
     into a structured dict and a fixed-format text brief. Always runs, free,
     reproducible. This is the source of truth.
  2. OPTIONAL LLM layer (Gemini): turns the structured summary into a natural-
     language narrative for a chosen audience. Enabled via config.LLM_ENABLED
     and a GEMINI_API_KEY env var. Falls back to the deterministic text if the
     key or SDK is unavailable, so the pipeline never breaks without an LLM.

The LLM only ever sees the already-computed numbers — it narrates, it does not
forecast. This keeps all the maths deterministic and auditable.
"""
from __future__ import annotations
import os
import json
import pandas as pd

from .. import config


def build_summary(state: dict) -> dict:
    """Collect headline numbers from the pipeline state dict."""
    s = {}
    if "data_report" in state:
        r = state["data_report"]
        s["data"] = {"rows": r["n_sales_rows"], "skus": r["n_skus"],
                     "stores": r["n_stores"], "span": f"{r['date_min']} → {r['date_max']}",
                     "clean": r["clean"]}
    if "classification" in state:
        s["classification"] = state["classification"]["Class"].value_counts().to_dict()
    if "validation" in state:
        s["validation"] = {"passed": state["validation"]["passed"],
                            **state["validation"]["metrics"]}
    if "halo" in state and state["halo"].get("substitution_diagnostics"):
        s["substitution"] = state["halo"]["substitution_diagnostics"]
    if "halo" in state and state["halo"].get("routing_decision"):
        rd = state["halo"]["routing_decision"]
        s["brand_routing"] = {
            "n_brand_helped": sum(1 for v in rd.values() if v["brand_helped"]),
            "n_total": len(rd),
            "detail": rd,
        }
    if "halo" in state and state["halo"].get("anchor_diagnostics"):
        s["anchor_ml"] = state["halo"]["anchor_diagnostics"]
    if "reconciliation" in state:
        s["reconciliation"] = state["reconciliation"]["report"]
    if "inventory" in state:
        s["inventory_scenarios"] = state["inventory"]["scenarios"].to_dict("records")
    if "npd" in state:
        s["cold_start"] = {k: state["npd"][k] for k in
                           ("analogs", "analog_prior", "blended_day1") if k in state["npd"]}
    return s


def render_text(summary: dict) -> str:
    lines = ["=" * 60, "LUMINA FORECASTING RUN — SUMMARY", "=" * 60]
    if "data" in summary:
        d = summary["data"]
        lines += [f"Data    : {d['rows']:,} rows · {d['skus']} SKUs · {d['stores']} stores",
                  f"          {d['span']} · clean={d['clean']}"]
    if "classification" in summary:
        cc = ", ".join(f"{k}={v}" for k, v in summary["classification"].items())
        lines.append(f"Demand  : {cc}")
    if "validation" in summary:
        v = summary["validation"]
        lines.append(f"Quality : passed={v.get('passed')} · WMAPE={v.get('wmape')} · "
                     f"interval_cov={v.get('interval_coverage')}")
    if "substitution" in summary:
        lines.append("Halo    :")
        for name, diag in summary["substitution"].items():
            if diag:
                lines.append(f"          {name}: premium {diag['premium_baseline']} → "
                             f"{diag['premium_when_rival_promo']} when rival promo "
                             f"({diag['lift_pct']}%)")
    if "brand_routing" in summary:
        br = summary["brand_routing"]
        lines.append(f"Brand   : {br['n_brand_helped']}/{br['n_total']} halo SKUs forecast better at "
                     f"brand level (escalated); rest kept at SKU level")
    if "anchor_ml" in summary:
        lines.append("AnchorML:")
        for name, info in summary["anchor_ml"].items():
            lines.append(f"          {name}: {info['anchor_sku']}→{info['attach_sku']} · "
                         f"anchor signal rank {info['anchor_feature_rank']}/{info['n_features']} "
                         f"in attach model")
    if "reconciliation" in summary:
        r = summary["reconciliation"]
        method = r.get("method", "mint")
        line = f"MinT/LP : [{method}] pre-gap {r['pre_gap_pct']:.1f}% → post-gap {r['post_gap']:.2e}"
        if method == "l1_lp":
            cap_drop = r["incoherent_total"] - r["reconciled_total"]
            line += f"\n          DC cap bound: total {r['incoherent_total']:.0f} → {r['reconciled_total']:.0f} (−{cap_drop:.0f})"
            if "units_added_by_rounding" in r:
                line += f"\n          batch rounding added {r['units_added_by_rounding']:.0f} units"
        lines.append(line)
    if "inventory_scenarios" in summary:
        lines.append("Inventory:")
        for sc in summary["inventory_scenarios"]:
            lines.append(f"          {sc['scenario']}: CR={sc['critical_ratio']} → {sc['order_quantile']}")
    if "cold_start" in summary:
        c = summary["cold_start"]
        lines.append(f"ColdStart: analogs={c.get('analogs')} · day-1={c.get('blended_day1')}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _audience_directive(audience: str) -> str:
    """Map an audience preset to a style instruction for the LLM."""
    return {
        "executive": ("Write for a busy executive. 4-6 sentences of flowing prose, "
                      "no jargon, lead with the business takeaway. State whether the "
                      "forecasts are trustworthy and what they enable."),
        "analyst": ("Write for a data analyst. Explain what each metric means and "
                    "whether it is good or bad, referencing WMAPE, coverage, and the "
                    "reconciliation gap. Be precise and quantitative."),
        "planner": ("Write for a supply-chain planner. Focus on the inventory "
                    "implications: which products to over- vs under-stock and why, "
                    "and any capacity constraints that bound the plan."),
    }.get(audience, "Write a clear, concise summary.")


def build_llm_prompt(summary: dict, audience: str = "executive") -> str:
    """Construct the prompt: a system-style directive + the structured numbers.

    The LLM is given ONLY the already-computed summary dict (as JSON) and asked
    to narrate it. It must not invent numbers — every figure it cites must come
    from the provided data.
    """
    directive = _audience_directive(audience)
    facts = json.dumps(summary, indent=2, default=str)
    return (
        "You are a reporting assistant for a retail demand-forecasting system. "
        "Summarise the run results below in natural language.\n\n"
        f"STYLE: {directive}\n\n"
        "RULES:\n"
        "- Use ONLY the numbers in the data. Do not invent or estimate anything.\n"
        "- If a figure is missing, simply omit it; never guess.\n"
        "- Do not output bullet points or headings unless the style asks for them.\n\n"
        f"RUN DATA (JSON):\n{facts}\n\n"
        "Now write the summary:"
    )


def _call_gemini(prompt: str) -> str:
    """Call the real Gemini API via either AI Studio or Vertex/GCP.

    Backend is selected by config.LLM_BACKEND:
      - "aistudio": needs an API key in $GEMINI_API_KEY.
      - "vertex"  : needs a GCP project (in $GOOGLE_CLOUD_PROJECT) and
                    Application Default Credentials (`gcloud auth ...`).
    Raises on any failure so the caller can fall back to the deterministic text.
    """
    # google-genai is the current unified SDK (`pip install google-genai`)
    from google import genai
    from google.genai import types

    if config.LLM_BACKEND == "vertex":
        project = os.environ.get(config.GCP_PROJECT_ENV)
        if not project:
            raise RuntimeError(f"No GCP project in ${config.GCP_PROJECT_ENV}")
        # Auth comes from Application Default Credentials (no API key).
        client = genai.Client(vertexai=True, project=project,
                              location=config.GCP_LOCATION)
    else:  # "aistudio"
        api_key = os.environ.get(config.LLM_API_KEY_ENV)
        if not api_key:
            raise RuntimeError(f"No API key in ${config.LLM_API_KEY_ENV}")
        client = genai.Client(api_key=api_key)

    resp = client.models.generate_content(
        model=config.LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=config.LLM_TEMPERATURE,
            max_output_tokens=config.LLM_MAX_TOKENS,
        ),
    )
    return resp.text.strip()


def narrate(summary: dict,
            audience: str | None = None,
            client=None) -> dict:
    """Produce a natural-language narrative of the summary.

    Parameters
    ----------
    client : optional callable(prompt:str) -> str. If given, it is used instead
             of the real Gemini call — handy for testing/mocking. If None, the
             real Gemini API is used.

    Returns {available: bool, text: str|None, error: str|None}.
    """
    audience = audience or config.LLM_AUDIENCE
    prompt = build_llm_prompt(summary, audience)
    try:
        caller = client if client is not None else _call_gemini
        text = caller(prompt)
        return {"available": True, "text": text, "error": None}
    except Exception as e:  # missing key, SDK, network — degrade gracefully
        return {"available": False, "text": None, "error": str(e)}


def run(state: dict, llm_client=None) -> dict:
    """Entry point.

    Always returns the deterministic summary + text. If config.LLM_ENABLED is
    True (or an llm_client is supplied for testing), also returns an LLM
    narrative under 'llm_text'. `combined` stitches them: deterministic brief
    first, natural-language narrative second.
    """
    summary = build_summary(state)
    text = render_text(summary)
    out = {"summary": summary, "text": text, "llm_text": None, "llm_error": None}

    if config.LLM_ENABLED or llm_client is not None:
        nl = narrate(summary, client=llm_client)
        out["llm_text"] = nl["text"]
        out["llm_error"] = nl["error"]

    # combined view: deterministic first, NL second
    combined = text
    if out["llm_text"]:
        combined += "\n\n" + "-" * 60 + "\nNATURAL-LANGUAGE SUMMARY\n" + "-" * 60 + \
                    "\n" + out["llm_text"]
    out["combined"] = combined
    return out


if __name__ == "__main__":
    # Demonstration with a MOCK Gemini client (no API key / network needed).
    # The mock mimics the shape of a real Gemini narrative so you can see the
    # end-to-end behaviour. Swap llm_client=None to use the real API once
    # GEMINI_API_KEY is set and config.LLM_ENABLED = True.

    def mock_gemini(prompt: str) -> str:
        # A real call would send `prompt` to Gemini; here we return a canned
        # narrative consistent with the demo numbers to illustrate the output.
        return (
            "This forecasting run is healthy and its outputs can be trusted. "
            "Across 387,430 daily records spanning 53 products and 10 stores, "
            "the model's median forecasts were off by only about 15% of total "
            "volume, and its uncertainty ranges captured roughly 84% of actual "
            "sales — close to the 80% target, meaning the stated confidence is "
            "honest. The system correctly handled cross-product effects: it "
            "confirmed that promoting private-label soda cuts premium soda sales "
            "by 45%, and for three of eight cross-product items it improved "
            "accuracy by forecasting at the brand level rather than the "
            "individual product. After reconciliation the forecasts add up "
            "perfectly across the store-to-region hierarchy. In practical terms, "
            "staples should be stocked generously to avoid stockouts while "
            "perishables are held closer to expected demand to limit spoilage."
        )

    # Build a representative summary dict (mirrors a real run's structure).
    demo_summary = {
        "data": {"rows": 387430, "skus": 53, "stores": 10,
                 "span": "2024-01-01 → 2025-12-31", "clean": True},
        "classification": {"Smooth": 27, "Lumpy": 25, "Erratic": 1},
        "validation": {"passed": True, "wmape": 0.151, "interval_coverage": 0.843},
        "substitution": {"SODA": {"premium_baseline": 403.2,
                                  "premium_when_rival_promo": 220.4, "lift_pct": -45.3}},
        "brand_routing": {"n_brand_helped": 3, "n_total": 8},
        "reconciliation": {"method": "mint", "pre_gap_pct": 16.8, "post_gap": 1.8e-12},
        "inventory_scenarios": [
            {"scenario": "staple", "critical_ratio": 0.952, "order_quantile": "P90"},
            {"scenario": "perishable", "critical_ratio": 0.545, "order_quantile": "P50"},
        ],
    }

    print(render_text(demo_summary))
    print()
    nl = narrate(demo_summary, audience="executive", client=mock_gemini)
    print("-" * 60)
    print("NATURAL-LANGUAGE SUMMARY  (audience: executive)")
    print("-" * 60)
    print(nl["text"])
