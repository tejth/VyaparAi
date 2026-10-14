"""Settings and model status: data source, quality, AI connectivity, forecast model, maintenance."""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Settings and Model Status", "What the app is running on, which modes are degraded, and maintenance tools.")

last = ctx.repo.last_analysis_run(ctx.active)
fm = ctx.repo.forecast_models_used(ctx.active)
mode, detail = state.llm_mode_summary()
statuses = {s.backend: s for s in state.llm_statuses()}

st.subheader("Status")
C.kpis([
    ("Data source", "Demo (synthetic)" if ctx.active == "demo" else "Uploaded", None),
    ("Data quality", ctx.ds.quality_status(), "Details in Data Management."),
    ("Records processed", f"{ctx.ds.record_count():,}", f"{len(ctx.ds.sales):,} sales · {len(ctx.ds.inventory):,} inventory · {len(ctx.ds.purchases):,} purchase rows"),
    ("Last analysis", last["ran_at"].replace("T", " ") if last else "n/a", "Analyses are cached and only rerun when data or thresholds change."),
], per_row=4)

st.subheader("AI model (free, runs locally)")
rows = []
for b in ("ollama", "huggingface"):
    s = statuses[b]
    rows.append({"Backend": b, "Model": s.model, "Available": "Yes" if s.available else "No", "Details": s.message})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.markdown(f"**Current AI mode:** {mode}")
backend = st.selectbox("Language-model backend", ["auto", "ollama", "huggingface", "fallback"],
                       index=["auto", "ollama", "huggingface", "fallback"].index(state.current_backend()),
                       format_func=lambda b: {"auto": "Auto (Ollama, then Hugging Face, then fallback)", "ollama": "Ollama only", "huggingface": "Hugging Face local only",
                                              "fallback": "Fallback only (no language model)"}[b], key="set_backend")
if backend != state.current_backend():
    st.session_state["llm_backend"] = backend
    st.rerun()
if st.button("Re-check model status", key="set_recheck"):
    state.llm_statuses.clear()
    st.rerun()
st.caption(f"Ollama host `{config.OLLAMA_HOST}` · model `{config.OLLAMA_MODEL}` · Hugging Face model `{config.HF_MODEL}`. Change them in the `.env` file. "
           "No API key, account or payment is needed for any of these.")

st.subheader("Forecast model")
if fm.empty:
    st.caption("No forecasts have been trained yet. Open Demand Forecasting, or run forecasts for all products from Inventory Intelligence.")
else:
    latest = fm.drop_duplicates(["product_id", "horizon"])
    mix = latest["model"].map(lambda m: {"recent_average": "Recent average", "seasonal_weekly": "Weekly seasonal baseline", "random_forest": "Random Forest (ML)"}.get(m, m)).value_counts()
    st.markdown("Models selected on validation so far: " + ", ".join(f"**{k}**: {v}" for k, v in mix.items()))
    show = latest.head(15).rename(columns={"product_id": "Product", "horizon": "Horizon (days)", "model": "Selected model", "mae": "MAE", "wape": "WAPE",
                                           "baseline_mae": "Baseline MAE", "baseline_wape": "Baseline WAPE", "trained_at": "Trained at"}).drop(columns=["dataset", "data_version"])
    st.dataframe(show, hide_index=True, width="stretch")
    st.caption("The ML model is only used when it beats the baselines on time-based validation.")

st.subheader("Degraded or fallback modes")
issues = []
if not mode.startswith("AI"):
    issues.append(f"AI Business Analyst is in **fallback mode**: {detail}")
if ctx.ds.inventory.empty:
    issues.append("No inventory file: inventory intelligence is unavailable.")
if ctx.ds.purchases.empty:
    issues.append("No purchases file: stock on order and reconciliation are unavailable.")
if ctx.ds.sales["cost_missing"].any():
    issues.append(f"{int(ctx.ds.sales['cost_missing'].sum())} sales lines have no recorded unit cost (estimated or excluded from margins).")
if ctx.analysis.inventory["lead_time_assumed"].any() if "lead_time_assumed" in ctx.analysis.inventory else False:
    issues.append("Some supplier lead times were assumed (7 days).")
if (ctx.ds.as_of < pd.Timestamp.today().normalize() - pd.Timedelta(days=45)) and ctx.active != "demo":
    issues.append(f"The latest sale is {ctx.ds.as_of:%d %b %Y}; the data may be out of date.")
if issues:
    for i in issues:
        st.warning(i)
else:
    st.success("No degraded modes.")

st.subheader("Business rules")
st.caption("These thresholds drive inventory classification and anomaly detection. They can also be edited on the Inventory and Profit Leakage pages.")
C.rules_editor(ctx, list(C.RULE_META), "set_rules")

st.subheader("Database")
st.code(f"Location: {config.DB_PATH}", language="text")
b1, b2 = st.columns(2)
if b1.button("Create backup", key="set_backup"):
    dest = config.DATA_DIR / "backups" / f"vyaparai_{datetime.now():%Y%m%d_%H%M%S}.db"
    ctx.repo.backup(dest)
    st.success(f"Backup saved to {dest}")
with b2:
    if st.button("Erase ALL stored data", key="set_erase"):
        st.session_state["confirm_erase"] = True
    if st.session_state.get("confirm_erase"):
        st.error("This deletes the demo data, your uploaded data, settings and history from the local database. Original CSV files are not touched. The demo data is restored automatically.")
        y, n = st.columns(2)
        if y.button("Yes, erase", key="set_erase_yes", type="primary"):
            ctx.repo.reset_database()
            st.session_state["confirm_erase"] = False
            st.cache_resource.clear()
            st.rerun()
        if n.button("Cancel", key="set_erase_no"):
            st.session_state["confirm_erase"] = False
            st.rerun()
st.caption("Everything stays on this computer. No data is sent to any cloud service.")
