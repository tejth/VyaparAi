"""Sales analytics: trends, rankings, categories, discounts, period comparison, export."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics import profitability as P  # noqa: E402
from src.analytics import sales as S  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402
from src.utils.formatting import inr, inr_short, num, pct  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Sales Analytics", "Trends, rankings, discounts and period comparisons from your transactions.")
C.demo_banner(ctx)
if ctx.ds.is_empty:
    theme.empty_state("No valid sales data is loaded. Go to <b>Data Management</b>.")
    st.stop()

start, end, cats = C.date_category_filters(ctx, "sa")
sales = P.filter_sales(ctx.ds.sales, start, end, cats or None)
if sales.empty:
    theme.empty_state("No sales match the selected filters.")
    st.stop()
s = P.compute_summary(sales)
C.kpis([("Revenue", inr_short(s.revenue), inr(s.revenue)), ("Gross profit", inr_short(s.gross_profit), "Not net profit."),
        ("Gross margin", pct(s.gross_margin_pct), None), ("Average selling price", inr(s.avg_selling_price, 2), "Revenue / units on sale lines."),
        ("Discounts given", inr_short(s.discount_value), f"{pct(s.discount_pct_of_gross)} of list-price value.")])
C.status_caption(ctx)

tab_trend, tab_prod, tab_cat, tab_disc, tab_cmp, tab_export = st.tabs(
    ["Trends", "Products", "Categories", "Discounts", "Period comparison", "Export"])

with tab_trend:
    freq = st.radio("Group by", ["Daily", "Weekly", "Monthly"], index=1, horizontal=True, key="sa_freq")
    ts_all = P.timeseries(sales, {"Daily": "D", "Weekly": "W", "Monthly": "M"}[freq])
    ts = P.complete_periods(ts_all)
    if len(ts) < len(ts_all):
        st.caption("Partial first/last weeks or months are hidden so they do not look like a drop in sales.")
    rev, unit, suf = theme.scale_money(ts["revenue"])
    k = (ts["revenue"].abs().max() / rev.abs().max()) if rev.abs().max() else 1
    f = go.Figure()
    f.add_scatter(x=ts["period"], y=rev, name="Revenue", mode="lines", line=dict(color=theme.C_REVENUE, width=2), hovertemplate=f"%{{y:,.2f}}{suf}")
    f.add_scatter(x=ts["period"], y=ts["gross_profit"] / k, name="Gross profit", mode="lines", line=dict(color=theme.C_PROFIT, width=2),
                  hovertemplate=f"%{{y:,.2f}}{suf}")
    f.update_yaxes(title=unit)
    st.plotly_chart(theme.style_fig(f, 380).update_layout(title="Revenue and gross profit (same scale)"), width="stretch")
    f = go.Figure(go.Bar(x=ts["period"], y=ts["units"], marker=dict(color=theme.C_REVENUE, cornerradius=3), name="Units"))
    st.plotly_chart(theme.style_fig(f, 260, legend=False).update_layout(title="Units sold"), width="stretch")
    st.caption(S.correlation_note())

prod = P.by_product(sales)
with tab_prod:
    c1, c2, c3 = st.columns(3)
    metric = c1.selectbox("Rank by", ["revenue", "gross_profit", "units", "margin_pct"], format_func=lambda x: {
        "revenue": "Revenue", "gross_profit": "Gross profit", "units": "Units", "margin_pct": "Gross margin %"}[x], key="sa_rank")
    top_n = c2.slider("Show top", 5, 30, 12, key="sa_topn")
    order = c3.radio("Order", ["Highest", "Lowest"], horizontal=True, key="sa_order")
    ranked = prod.dropna(subset=[metric]).sort_values(metric, ascending=(order == "Lowest")).head(top_n).iloc[::-1]
    f = go.Figure(go.Bar(x=ranked[metric], y=ranked["product_name"], orientation="h", marker=dict(color=theme.C_REVENUE, cornerradius=4)))
    st.plotly_chart(theme.style_fig(f, 90 + 26 * len(ranked), legend=False).update_layout(yaxis=dict(automargin=True)), width="stretch")

    st.markdown("**Quantity versus revenue** (each dot is a product; dot size = gross profit magnitude)")
    sc = prod[prod["revenue"] > 0].copy()
    sc["size"] = sc["gross_profit"].abs().clip(lower=1)
    f = go.Figure(go.Scatter(x=sc["units"], y=sc["revenue"], mode="markers", text=sc["product_name"],
                             marker=dict(color=theme.C_REVENUE, size=(sc["size"] ** 0.5).clip(upper=40).clip(lower=6), opacity=0.8, line=dict(color=theme.SURFACE, width=1.5)),
                             hovertemplate="%{text}<br>%{x:,.0f} units<br>₹%{y:,.0f}<extra></extra>"))
    f.update_xaxes(title="Units sold", type="log")
    f.update_yaxes(title="Revenue (₹)", type="log")
    st.plotly_chart(theme.style_fig(f, 420, legend=False), width="stretch")
    st.dataframe(prod.rename(columns={"product_name": "Product", "category": "Category", "units": "Units", "revenue": "Revenue", "cogs": "COGS",
                                      "gross_profit": "Gross profit", "margin_pct": "Margin", "avg_price": "Avg price", "avg_discount_pct": "Avg discount"}).drop(columns=["product_id", "cost_coverage_pct", "transactions"]),
                 hide_index=True, width="stretch",
                 column_config={"Revenue": C.money_cfg("Revenue (₹)"), "COGS": C.money_cfg("COGS (₹)"), "Gross profit": C.money_cfg("Gross profit (₹)"),
                                "Margin": C.pct_cfg("Margin"), "Avg price": C.money2_cfg("Avg price (₹)"), "Avg discount": C.pct_cfg("Avg discount"), "Units": C.num_cfg("Units")})

with tab_cat:
    cat = P.by_category(sales)
    f = go.Figure(go.Bar(x=cat["margin_pct"], y=cat["category"], orientation="h", marker=dict(color=theme.C_PROFIT, cornerradius=4),
                         hovertemplate="%{y}: %{x:.1f}%<extra></extra>"))
    st.plotly_chart(theme.style_fig(f, 360, legend=False).update_layout(title="Gross margin % by category", yaxis=dict(autorange="reversed", automargin=True)), width="stretch")
    st.dataframe(cat, hide_index=True, width="stretch", column_config={
        "revenue": C.money_cfg("Revenue (₹)"), "cogs": C.money_cfg("COGS (₹)"), "gross_profit": C.money_cfg("Gross profit (₹)"),
        "margin_pct": C.pct_cfg("Margin"), "revenue_share_pct": C.pct_cfg("Revenue share")})

with tab_disc:
    d = S.discount_analysis(sales)
    if d["by_product"].empty:
        st.info("No sale lines to analyse.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            f = go.Figure(go.Bar(x=d["distribution"]["discount_band"], y=d["distribution"]["lines"], marker=dict(color=theme.C_COST, cornerradius=4)))
            st.plotly_chart(theme.style_fig(f, 320, legend=False).update_layout(title="Sale lines by discount band"), width="stretch")
        with c2:
            m = d["by_month"]
            f = go.Figure(go.Scatter(x=m["month"], y=m["discount_pct"], mode="lines+markers", line=dict(color=theme.C_COST, width=2), marker=dict(size=8)))
            f.update_yaxes(title="% of list-price value")
            st.plotly_chart(theme.style_fig(f, 320, legend=False).update_layout(title="Average discount depth by month"), width="stretch")
        st.markdown("**Products with the most discount given away**")
        st.dataframe(d["by_product"].head(15).drop(columns=["product_id"]).rename(columns={"product_name": "Product"}), hide_index=True, width="stretch",
                     column_config={"gross_sales": C.money_cfg("List value (₹)"), "discount_value": C.money_cfg("Discount given (₹)"),
                                    "discount_pct": C.pct_cfg("Discount %"), "discounted_share_pct": C.pct_cfg("Lines discounted")})
        st.caption("Discount % = discount value / list-price value (quantity x unit price). Heavy discounting on thin-margin products is the usual way margin is lost.")

with tab_cmp:
    window = st.radio("Compare the latest", [7, 14, 30, 60, 90], index=2, horizontal=True, format_func=lambda d: f"{d} days", key="sa_win")
    cmp = S.period_comparison(ctx.ds.sales, ctx.ds.as_of, window)
    st.caption(f"Latest: {cmp.current_label} · Previous: {cmp.previous_label}")
    if not cmp.sufficient:
        st.warning(cmp.note)
    else:
        m = cmp.metrics.copy()
        st.dataframe(m.rename(columns={"metric": "Metric", "current": "Latest", "previous": "Previous", "change": "Change", "change_pct": "Change %"}),
                     hide_index=True, width="stretch", column_config={"Change %": C.pct_cfg("Change %"),
                                                                      "Latest": st.column_config.NumberColumn(format="%.1f"), "Previous": st.column_config.NumberColumn(format="%.1f"),
                                                                      "Change": st.column_config.NumberColumn(format="%.1f")})
        st.caption(cmp.note)
        dec = S.declining_products(ctx.ds.sales, ctx.ds.as_of, window, 30.0)
        st.markdown(f"**Products selling at least 30% fewer units than in the previous {window} days** ({len(dec)})")
        if dec.empty:
            st.success("No products show a sustained decline.")
        else:
            st.dataframe(dec.drop(columns=["product_id"]).rename(columns={"product_name": "Product", "category": "Category", "units_prev": "Units before", "units_cur": "Units now",
                                                                         "change_pct": "Change", "revenue_cur": "Revenue now", "revenue_prev": "Revenue before"}),
                         hide_index=True, width="stretch", column_config={"Change": C.pct_cfg("Change"), "Revenue now": C.money_cfg("Revenue now (₹)"),
                                                                          "Revenue before": C.money_cfg("Revenue before (₹)")})

with tab_export:
    st.markdown("Downloads reflect the current date and category filters.")
    c1, c2, c3 = st.columns(3)
    with c1:
        C.csv_button(prod, "vyaparai_product_performance.csv", "Product performance (CSV)", key="ex_prod")
    with c2:
        C.csv_button(P.timeseries(sales, "D"), "vyaparai_daily_sales.csv", "Daily revenue and profit (CSV)", key="ex_ts")
    with c3:
        C.csv_button(sales.drop(columns=["unit_cost_eff", "cost_source"], errors="ignore"), "vyaparai_filtered_transactions.csv", "Filtered transactions (CSV)", key="ex_tx")
