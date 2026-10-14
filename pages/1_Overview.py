"""Executive overview dashboard."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

import config  # noqa: E402
from src.analytics import inventory as I  # noqa: E402
from src.analytics import profitability as P  # noqa: E402
from src.analytics import recommendations as R  # noqa: E402
from src.ml import anomaly_detection as A  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402
from src.utils.formatting import inr, inr_short, num, pct  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Overview", config.TAGLINE)
C.demo_banner(ctx)

if ctx.ds.is_empty:
    theme.empty_state("No valid sales data is loaded. Go to <b>Data Management</b> to load the demo data or upload your CSV files.")
    st.stop()

start, end, cats = C.date_category_filters(ctx, "ov")
sales = P.filter_sales(ctx.ds.sales, start, end, cats or None)
if sales.empty:
    theme.empty_state("No sales match the selected filters.")
    st.stop()

s = P.compute_summary(sales)
inv = C.filter_inventory(ctx, cats)
ic = I.summary_counts(inv)
anoms = C.filter_anomalies(ctx, start, end, cats)
ac = A.summarize(anoms)

C.kpis([
    ("Total revenue", inr_short(s.revenue), f"Recognised revenue = recorded line totals net of discount, returns negative. Exact: {inr(s.revenue)}"),
    ("Cost of goods sold", inr_short(s.cogs), "Quantity x unit cost for lines with a known cost (recorded, or estimated from inventory cost)."),
    ("Gross profit", inr_short(s.gross_profit), "Revenue minus cost of goods sold. NOT net profit: rent, salaries and other overheads are not in the data."),
    ("Gross margin", pct(s.gross_margin_pct), "Gross profit / revenue on lines with a known cost x 100."),
    ("Units sold", num(s.units_sold), "Net of returns."),
    ("Active products", num(s.active_products), "Products with at least one sale in the selected period."),
    ("Low-stock products", num(ic["low_stock"]), "Stocked out, at/below the reorder point, or at risk during supplier lead time (current snapshot)."),
    ("Stock value (at cost)", inr_short(ic["inventory_value"]), f"Stock x unit cost. Cost known for {pct(ic['inventory_value_known_pct'], 0)} of products."),
    ("Slow-moving products", num(ic["slow_moving"]), "Slow-moving plus dead stock, using the thresholds in Settings."),
    ("Detected anomalies", num(ac["total"]), f"{ac['high']} high severity. Findings need review; they are not proof of wrongdoing."),
])
C.status_caption(ctx)

# ---- trends
st.subheader("Revenue and gross profit over time")
freq_label = st.radio("Group by", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="ov_freq")
ts_all = P.timeseries(sales, {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq_label])
ts = P.complete_periods(ts_all)
if len(ts) < len(ts_all):
    st.caption(f"Partial first/last {freq_label.lower()[:-2]} periods are hidden so they do not look like a drop in sales.")
rev, unit, suffix = theme.scale_money(ts["revenue"])
c1, c2 = st.columns(2)
with c1:
    f = go.Figure(go.Scatter(x=ts["period"], y=rev, mode="lines", line=dict(color=theme.C_REVENUE, width=2), name="Revenue",
                             hovertemplate=f"%{{y:,.2f}}{suffix}"))
    f.update_yaxes(title=unit)
    st.plotly_chart(theme.style_fig(f, 320, legend=False).update_layout(title="Revenue"), width="stretch")
with c2:
    g, gunit, gsuffix = theme.scale_money(ts["gross_profit"])
    f = go.Figure(go.Scatter(x=ts["period"], y=g, mode="lines", line=dict(color=theme.C_PROFIT, width=2), name="Gross profit",
                             hovertemplate=f"%{{y:,.2f}}{gsuffix}"))
    f.update_yaxes(title=gunit)
    st.plotly_chart(theme.style_fig(f, 320, legend=False).update_layout(title="Gross profit (not net profit)"), width="stretch")

# ---- products / categories
prod = P.by_product(sales)
b1, b2 = st.columns(2)
with b1:
    top = prod.sort_values("units", ascending=False).head(10).iloc[::-1]
    f = go.Figure(go.Bar(x=top["units"], y=top["product_name"], orientation="h", marker=dict(color=theme.C_REVENUE, cornerradius=4),
                         hovertemplate="%{y}: %{x:,.0f} units<extra></extra>"))
    st.plotly_chart(theme.style_fig(f, 380, legend=False).update_layout(title="Best-selling products (units)", yaxis=dict(automargin=True)), width="stretch")
with b2:
    top = prod.sort_values("gross_profit", ascending=False).head(10).iloc[::-1]
    f = go.Figure(go.Bar(x=top["gross_profit"], y=top["product_name"], orientation="h", marker=dict(color=theme.C_PROFIT, cornerradius=4),
                         hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
    st.plotly_chart(theme.style_fig(f, 380, legend=False).update_layout(title="Highest gross-profit products (₹)", yaxis=dict(automargin=True)), width="stretch")

st.subheader("Category performance")
cat = P.by_category(sales)
f = go.Figure()
rv, runit, _ = theme.scale_money(cat["revenue"])
f.add_bar(x=cat["category"], y=rv, name="Revenue", marker=dict(color=theme.C_REVENUE, cornerradius=4), hovertemplate="%{x}: %{y:,.2f}<extra>Revenue</extra>")
gscale = (cat["revenue"].abs().max() / rv.abs().max()) if rv.abs().max() else 1
f.add_bar(x=cat["category"], y=cat["gross_profit"] / gscale, name="Gross profit", marker=dict(color=theme.C_PROFIT, cornerradius=4),
          hovertemplate="%{x}: %{y:,.2f}<extra>Gross profit</extra>")
f.update_layout(barmode="group", bargap=0.25)
f.update_yaxes(title=runit)
st.plotly_chart(theme.style_fig(f, 360), width="stretch")
st.dataframe(cat.rename(columns={"category": "Category", "units": "Units", "revenue": "Revenue", "cogs": "COGS", "gross_profit": "Gross profit",
                                 "margin_pct": "Margin", "revenue_share_pct": "Revenue share", "products": "Products"}),
             hide_index=True, width="stretch",
             column_config={"Revenue": C.money_cfg("Revenue (₹)"), "COGS": C.money_cfg("COGS (₹)"), "Gross profit": C.money_cfg("Gross profit (₹)"),
                            "Margin": C.pct_cfg("Margin"), "Revenue share": C.pct_cfg("Revenue share"), "Units": C.num_cfg("Units")})

# ---- alerts and recommendations
left, right = st.columns([1, 1])
with left:
    st.subheader("Recent alerts")
    if anoms.empty:
        st.success("No anomalies detected for the selected filters.")
    else:
        for r in anoms.head(8).itertuples():
            st.markdown(f'<div class="vy-card">{theme.severity_label(r.severity)} &nbsp; <b>{r.detection}</b> · {r.product_name}'
                        f'<div class="vy-muted">{pd.Timestamp(r.date):%d %b %Y} · {r.group} · observed {r.observed} (expected {r.expected})</div></div>',
                        unsafe_allow_html=True)
        st.caption(f"{len(anoms)} findings in total. Open **Profit Leakage Detection** for the full list and next steps.")
with right:
    st.subheader("Recommended actions")
    recs = ctx.analysis.recommendations
    st.caption(f"Ranked by priority, then estimated exposure. Total exposure counted once across overlapping items: "
               f"**{inr(R.total_counted_exposure(recs))}** (an estimate, not a promised saving).")
    for i, rec in enumerate(recs[:6]):
        C.recommendation_card(rec, i)
