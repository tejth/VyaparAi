"""Profit leakage detection: data-quality problems, statistical anomalies and business risks, kept separate."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics import profitability as P  # noqa: E402
from src.analytics import recommendations as R  # noqa: E402
from src.ml import anomaly_detection as A  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402
from src.utils.formatting import inr, inr_short, num, pct  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Profit Leakage Detection", "Where money may be leaking, and which records need a second look.")
C.demo_banner(ctx)
if ctx.ds.is_empty:
    theme.empty_state("No valid sales data is loaded. Go to <b>Data Management</b>.")
    st.stop()

st.info("Findings are **potential risks that require review**. A statistical anomaly can be a perfectly legitimate sale (bulk order, one-off price). "
        "Nothing here is evidence of fraud or theft.", icon="🔎")

start, end, cats = C.date_category_filters(ctx, "pl")
an = C.filter_anomalies(ctx, start, end, cats)
counts = A.summarize(an)
recs = ctx.analysis.recommendations
C.kpis([("Findings", num(counts["total"]), "Within the selected filters."), ("High severity", num(counts["high"]), None),
        ("Data-quality problems", num(counts["data_quality"]), "Verifiable problems in the records."),
        ("Statistical anomalies", num(counts["statistical"]), "Unusual versus the product's own history. Needs review."),
        ("Business risks", num(counts["business_risk"]), "Patterns that may erode profit.")])
st.caption(f"Estimated exposure across all recommendations, overlapping items counted once: **{inr(R.total_counted_exposure(recs))}**. "
           "This is an estimate of money at risk, not a promised saving.")

tab_find, tab_margin, tab_disc, tab_actions, tab_cfg = st.tabs(["Flagged records", "Margins", "Discount leakage", "Recommended actions", "Detection settings"])

with tab_find:
    if an.empty:
        st.success("No findings for the selected filters.")
    else:
        c1, c2, c3 = st.columns(3)
        groups = c1.multiselect("Group", ["Data quality", "Statistical anomaly", "Business risk"], default=["Data quality", "Statistical anomaly", "Business risk"], key="pl_g")
        sevs = c2.multiselect("Severity", ["High", "Medium", "Low"], default=["High", "Medium"], key="pl_s")
        kinds = c3.multiselect("Detection type", sorted(an["detection"].unique()), key="pl_k")
        view = an[an["group"].isin(groups) & an["severity"].isin(sevs) & (an["detection"].isin(kinds) if kinds else True)]
        by = view.groupby(["detection", "severity"]).size().reset_index(name="n")
        if len(by):
            f = go.Figure()
            for sev, color in (("High", "#d03b3b"), ("Medium", "#ec835a"), ("Low", theme.INK_MUTED)):
                d = by[by["severity"] == sev]
                f.add_bar(x=d["n"], y=d["detection"], orientation="h", name=sev, marker=dict(color=color, cornerradius=3))
            f.update_layout(barmode="stack", bargap=0.3)
            st.plotly_chart(theme.style_fig(f, 90 + 34 * by["detection"].nunique()).update_layout(title="Findings by type and severity", yaxis=dict(automargin=True)), width="stretch")
        st.markdown(f"**{len(view)}** records shown")
        table = view.assign(date=pd.to_datetime(view["date"]).dt.strftime("%d %b %Y"))[
            ["severity", "group", "detection", "date", "product_name", "record_id", "observed", "expected", "explanation", "next_step", "amount_inr"]].rename(columns={
            "severity": "Severity", "group": "Group", "detection": "Detection", "date": "Date", "product_name": "Product", "record_id": "Record",
            "observed": "Observed", "expected": "Expected / threshold", "explanation": "Explanation", "next_step": "Recommended next step", "amount_inr": "Amount involved"})
        st.dataframe(table, hide_index=True, width="stretch", column_config={"Amount involved": C.money_cfg("Amount involved (₹)")})
        C.csv_button(view, "vyaparai_flagged_records.csv", "Download flagged records (CSV)", key="pl_dl")

with tab_margin:
    sales = P.filter_sales(ctx.ds.sales, start, end, cats or None)
    prod = P.by_product(sales)
    loss = P.loss_making_products(prod)
    st.markdown(f"**Loss-making products in the selected period: {len(loss)}**")
    if loss.empty:
        st.success("No product has a negative gross profit in this period.")
    else:
        st.dataframe(loss[["product_name", "category", "units", "revenue", "cogs", "gross_profit", "margin_pct", "avg_discount_pct"]].rename(columns={
            "product_name": "Product", "category": "Category", "units": "Units", "revenue": "Revenue", "cogs": "COGS", "gross_profit": "Gross profit",
            "margin_pct": "Margin", "avg_discount_pct": "Avg discount"}), hide_index=True, width="stretch",
            column_config={"Revenue": C.money_cfg("Revenue (₹)"), "COGS": C.money_cfg("COGS (₹)"), "Gross profit": C.money_cfg("Gross profit (₹)"),
                           "Margin": C.pct_cfg("Margin"), "Avg discount": C.pct_cfg("Avg discount")})
    st.markdown("**Margin movement: latest 30 days versus the previous 30 days**")
    drop = P.recent_margin_drop(ctx.ds.sales, ctx.ds.as_of)
    if drop.empty:
        st.info("Not enough history to compare two 30-day periods.")
    else:
        worst = drop.head(10).iloc[::-1]
        f = go.Figure(go.Bar(x=worst["margin_change_pts"], y=worst["product_name"], orientation="h", marker=dict(color=theme.C_COST, cornerradius=4),
                             hovertemplate="%{y}: %{x:+.1f} points<extra></extra>"))
        f.update_xaxes(title="Change in gross margin (percentage points)")
        st.plotly_chart(theme.style_fig(f, 380, legend=False).update_layout(title="Largest margin declines", yaxis=dict(automargin=True)), width="stretch")
    st.caption("Gross margin uses recorded unit costs. A decline can come from lower prices, discounts, or a supplier cost increase; the data cannot say which.")
    miss = int(ctx.ds.sales["cost_missing"].sum())
    if miss:
        st.warning(f"{miss} sales line(s) have no recorded unit cost. Where possible the inventory cost is used as a flagged estimate; otherwise their margin is excluded rather than assumed to be zero.")

with tab_disc:
    from src.analytics import sales as S

    da = S.discount_analysis(P.filter_sales(ctx.ds.sales, start, end, cats or None))
    if da["by_product"].empty:
        st.info("No sale lines to analyse.")
    else:
        st.markdown("**Discount value given by product** (list price × quantity − amount charged)")
        top = da["by_product"].head(12).iloc[::-1]
        f = go.Figure(go.Bar(x=top["discount_value"], y=top["product_name"], orientation="h", marker=dict(color=theme.C_COST, cornerradius=4),
                             hovertemplate="%{y}: ₹%{x:,.0f}<extra></extra>"))
        st.plotly_chart(theme.style_fig(f, 400, legend=False).update_layout(yaxis=dict(automargin=True)), width="stretch")
        heavy = an[an["detection"].isin(["Unusually large discount", "Frequent heavy discounting"])]
        if len(heavy):
            st.markdown(f"**Heavy-discount findings:** {len(heavy)} (value {inr(heavy['amount_inr'].sum())})")

with tab_actions:
    st.caption("Ranked by priority, then estimated exposure. Each card lists evidence, assumptions and how the estimate was calculated.")
    if not recs:
        st.success("No actions needed right now.")
    for i, rec in enumerate(recs):
        C.recommendation_card(rec, i)
    C.csv_button(R.to_frame(recs).drop(columns=["counted"], errors="ignore"), "vyaparai_recommendations.csv", "Download recommendations (CSV)", key="pl_rec_dl")

with tab_cfg:
    st.write("Adjust how sensitive detection is. Changes are saved and all findings are recalculated.")
    C.rules_editor(ctx, ["high_discount_pct", "large_qty_zscore", "price_change_pct", "spike_zscore", "decline_pct", "low_margin_pct"], "pl_rules")
    st.markdown("""
**Rules and models used**
- *Data quality:* duplicates (same id, date, product, quantity, price and amount), totals that differ from quantity × price × (1 − discount), missing costs.
- *Statistical anomaly:* quantity far above the product's usual size, unit price far from its usual price, daily revenue spikes versus the same weekday,
  and an Isolation Forest that flags unusual combinations of quantity, price, discount and margin.
- *Business risk:* large or repeated heavy discounts, negative or low gross margin, margin declines, sudden sales declines.
""")
