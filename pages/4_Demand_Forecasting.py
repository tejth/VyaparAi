"""Demand forecasting: baselines vs ML, time-based validation, honest limitations."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.ml import forecasting as F  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402
from src.utils.formatting import num, pct  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Demand Forecasting", "Daily unit demand for the next 7 or 30 days, with accuracy checked against simple baselines.")
C.demo_banner(ctx)
if ctx.ds.is_empty:
    theme.empty_state("No valid sales data is loaded. Go to <b>Data Management</b>.")
    st.stop()

sales_prod = (ctx.ds.sales[~ctx.ds.sales["is_duplicate"] & ~ctx.ds.sales["is_return"]].groupby(["product_id", "product_name"])["quantity"].sum()
              .reset_index().sort_values("quantity", ascending=False))
labels = {r.product_id: f"{r.product_name} ({r.product_id})" for r in sales_prod.itertuples()}
c1, c2 = st.columns([3, 2])
pid = c1.selectbox("Product", list(labels), format_func=labels.get, key="fc_product")
horizon = c2.radio("Forecast horizon", [7, 30], horizontal=True, format_func=lambda h: f"Next {h} days", key="fc_h")

res = state.cached_forecast(ctx.ds, ctx.repo, ctx.active, ctx.version, pid, int(horizon))

for w in res.warnings:
    (st.error if not res.sufficient else st.warning)(w, icon="⚠️")
hist = res.history
if hist.empty:
    st.stop()

if res.sufficient:
    fc = res.forecast
    sel_label = F.MODEL_LABELS.get(res.selected_model, res.selected_model)
    mae_sel, wape_sel = res.metric(res.selected_model, "mae"), res.metric(res.selected_model, "wape")
    C.kpis([
        ("Expected units", num(res.expected_units), f"Estimate: sum of daily forecasts for {fc['date'].min():%d %b} to {fc['date'].max():%d %b %Y}."),
        ("Average per day", f"{res.avg_daily_forecast:.2f}", "Units per day (estimate)."),
        ("Validation MAE", f"{mae_sel:.2f}" if mae_sel is not None else "n/a", "Mean absolute error in units per day on held-out recent weeks."),
        ("Validation WAPE", pct(wape_sel * 100) if wape_sel is not None else "n/a", "Total absolute error / total actual demand. Safe when some days have zero sales (MAPE is not)."),
    ], per_row=4)
    st.markdown(f"**Model used:** {sel_label} · chosen on validation error; the ML model is used only if it clearly beats the baselines.")

    recent = hist.iloc[-90:]
    f = go.Figure()
    f.add_scatter(x=recent.index, y=recent.values, name="Actual daily demand", mode="lines", line=dict(color=theme.INK_MUTED, width=1.5))
    if fc["lower"].notna().any():
        f.add_scatter(x=pd.concat([fc["date"], fc["date"][::-1]]), y=pd.concat([fc["upper"], fc["lower"][::-1]]), fill="toself", name="Approx. 80% range",
                      mode="lines", line=dict(width=0), fillcolor="rgba(57,135,229,0.18)", hoverinfo="skip")
    f.add_scatter(x=fc["date"], y=fc["forecast"], name=f"Forecast ({sel_label})", mode="lines+markers", line=dict(color=theme.C_REVENUE, width=2.5), marker=dict(size=6))
    f.update_yaxes(title="Units per day", rangemode="tozero")
    st.plotly_chart(theme.style_fig(f, 400).update_layout(title="Last 90 days and forecast"), width="stretch")
    st.caption("The shaded range comes from forecast errors on recent validation weeks; real demand can fall outside it. Forecasts are estimates, not certainty.")

    st.subheader("ML versus baselines")
    if res.metrics.empty:
        st.info("A time-based validation was not possible with this much history, so only the recent-average estimate is shown.")
    else:
        m = res.metrics.copy()
        m["Model"] = m["model"].map(F.MODEL_LABELS)
        m["wape"] = m["wape"] * 100
        best = m["mae"].idxmin()
        m["Verdict"] = ["Selected" if r.selected else "" for r in m.itertuples()]
        st.dataframe(m[["Model", "mae", "wape", "rmse", "bias", "n", "Verdict"]].rename(columns={
            "mae": "MAE (units/day)", "wape": "WAPE", "rmse": "RMSE", "bias": "Bias (+ = over-forecast)", "n": "Days evaluated"}),
            hide_index=True, width="stretch", column_config={"WAPE": C.pct_cfg("WAPE"), "MAE (units/day)": C.num_cfg("MAE (units/day)", "%.3f"),
                                                             "RMSE": C.num_cfg("RMSE", "%.3f"), "Bias (+ = over-forecast)": C.num_cfg("Bias (+ = over-forecast)", "%.3f")})
        f = go.Figure()
        for key, col in (("recent_average", theme.PALETTE[3]), ("seasonal_weekly", theme.PALETTE[2]), ("random_forest", theme.PALETTE[0])):
            if key in fc.columns:
                f.add_scatter(x=fc["date"], y=fc[key], name=F.MODEL_LABELS[key], mode="lines+markers", line=dict(color=col, width=2), marker=dict(size=5))
        f.update_yaxes(title="Units per day", rangemode="tozero")
        st.plotly_chart(theme.style_fig(f, 320).update_layout(title="Forecasts from each model"), width="stretch")
        st.caption(f"Validation: {res.backtest_origins} rolling forecast origins in the most recent weeks, each predicting {horizon} days using only data available at that origin. "
                   "The ML model must beat the best baseline by at least 2% MAE to be selected.")
    C.csv_button(fc.assign(product=res.product_name), f"vyaparai_forecast_{pid}_{horizon}d.csv", "Download forecast (CSV)", key="fc_dl")
else:
    recent = hist.iloc[-90:]
    f = go.Figure(go.Scatter(x=recent.index, y=recent.values, mode="lines", line=dict(color=theme.INK_MUTED, width=1.5), name="Actual daily demand"))
    st.plotly_chart(theme.style_fig(f, 320).update_layout(title="Historical daily demand"), width="stretch")

st.subheader("Historical demand")
monthly = hist.resample("MS").sum()
f = go.Figure(go.Bar(x=monthly.index, y=monthly.values, marker=dict(color=theme.C_REVENUE, cornerradius=4), hovertemplate="%{x|%b %Y}: %{y:,.0f} units<extra></extra>"))
st.plotly_chart(theme.style_fig(f, 280, legend=False).update_layout(title="Units sold per month"), width="stretch")

with st.expander("How the forecast is built, and its limitations", expanded=False):
    st.markdown("""
**Baselines.** *Recent average* = mean of the last 28 days. *Weekly seasonal* = mean of the same weekday over the last 4 weeks.

**ML model.** A Random Forest (scikit-learn) predicts demand for each future day using only information known at the forecast origin:
the last 7 days of sales, 7/14/28-day averages, recent trend, how long since the last sale, plus calendar features of the target day
(weekday, month, day of year) and the forecast horizon. Predictions are never fed back in as inputs.

**Validation.** The most recent weeks are held out (never a random split). The model is trained only on days before that block and scored with rolling
forecast origins, exactly like the baselines. The final model is then refit on all data.

**Metrics.** MAE and WAPE; MAPE is avoided because many days have zero sales.
""")
    st.markdown("**Limitations**\n" + "\n".join(f"- {l}" for l in F.LIMITATIONS))
