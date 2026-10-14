"""Inventory intelligence: reorder points, stock-out risk, slow / dead / excess stock."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402

from src.analytics import inventory as I  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402
from src.utils.formatting import inr, inr_short, num, pct  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Inventory Intelligence", "What to reorder, what is tying up cash, and what is at risk of running out.")
C.demo_banner(ctx)
if ctx.ds.inventory.empty:
    theme.empty_state("No inventory file is loaded. Add one in <b>Data Management</b> to unlock inventory analysis.")
    st.stop()

cats = st.multiselect("Categories (empty = all)", sorted(ctx.ds.inventory["category"].dropna().unique()), key="inv_cats")
inv = C.filter_inventory(ctx, cats)
sc = I.summary_counts(inv)
rules = ctx.rules

C.kpis([
    ("Reorder or at risk", num(sc["low_stock"]), "Stocked out, at/below the reorder point, or short during lead time."),
    ("Stocked out", num(sc["stocked_out"]), "Zero stock for a product that normally sells."),
    ("Slow or dead stock", num(sc["slow_moving"]), f"{sc['dead_stock']} are dead stock (no sale for {rules.dead_stock_days_since_sale}+ days)."),
    ("Excess stock", num(sc["excess"]), f"More than {rules.excess_stock_days_cover} days of cover."),
    ("Stock value (at cost)", inr_short(sc["inventory_value"]), f"Exact: {inr(sc['inventory_value'])}. Cost known for {pct(sc['inventory_value_known_pct'], 0)} of products."),
])
tied = inv.loc[inv["status"].astype(str).isin(["Dead stock", "Slow-moving", "Excess stock"]), "inventory_value"].sum()
st.caption(f"Stock value in slow-moving, dead or excess products: **{inr(tied)}** at cost (working capital that is not turning over). "
           f"Demand source for these calculations: **{'ML forecast where it beat the baseline' if ctx.ml_demand else 'recent sales average'}**.")

with st.expander("Refine demand with ML forecasts (optional)"):
    st.write("By default average daily demand is the recent sales average, which is easy to explain. You can instead use each product's "
             "validated ML forecast, but only where the ML model beat the baselines on time-based validation. Training takes about a minute.")
    c1, c2 = st.columns([1, 3])
    if c1.button("Run forecasts for all products", key="inv_ml_run"):
        bar = st.progress(0.0, text="Training and validating models...")
        n = state.compute_ml_demand(ctx, progress=lambda f, name: bar.progress(f, text=f"Validated {name}"))
        st.session_state["use_ml_demand"] = True
        bar.empty()
        st.success(f"Done. ML demand is used for {n} product(s) where it beat the baseline; the rest keep the recent average.")
        st.rerun()
    if state.ml_demand_ready(ctx.active, ctx.version):
        st.session_state["use_ml_demand"] = c2.toggle("Use ML demand where available", value=bool(st.session_state.get("use_ml_demand")), key="inv_ml_toggle")

tab_alerts, tab_all, tab_promo, tab_how, tab_rec = st.tabs(["Alerts", "All products", "Promotion and margin candidates", "How it works", "Stock reconciliation"])

with tab_alerts:
    order = ["Stocked out", "Reorder now", "Stock-out risk", "Dead stock", "Slow-moving", "Excess stock"]
    counts = inv["status"].astype(str).value_counts().reindex(order + ["Healthy", "No demand data"]).fillna(0)
    f = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h", marker=dict(color=theme.C_REVENUE, cornerradius=4),
                         hovertemplate="%{y}: %{x:.0f} products<extra></extra>"))
    st.plotly_chart(theme.style_fig(f, 280, legend=False).update_layout(title="Products by stock status", yaxis=dict(autorange="reversed")), width="stretch")
    pick = st.multiselect("Show statuses", order, default=order[:3], key="inv_status")
    alerts = inv[inv["status"].astype(str).isin(pick)].copy()
    alerts["days_of_stock"] = alerts["days_of_stock"].replace(np.inf, np.nan)
    show = alerts[["product_name", "category", "status", "current_stock", "incoming_qty", "avg_daily_demand", "days_of_stock", "lead_time_days",
                   "reorder_point", "suggested_order_qty", "inventory_value", "confidence", "flags", "action"]].rename(columns={
        "product_name": "Product", "category": "Category", "status": "Status", "current_stock": "Stock", "incoming_qty": "On order",
        "avg_daily_demand": "Avg daily demand", "days_of_stock": "Days of stock left", "lead_time_days": "Lead time (days)", "reorder_point": "Reorder point",
        "suggested_order_qty": "Suggested order", "inventory_value": "Stock value", "confidence": "Confidence", "flags": "Flags", "action": "Recommended action"})
    st.dataframe(show, hide_index=True, width="stretch", column_config={
        "Stock": C.num_cfg("Stock"), "On order": C.num_cfg("On order"), "Avg daily demand": C.num_cfg("Avg daily demand", "%.2f"),
        "Days of stock left": C.num_cfg("Days of stock left", "%.1f"), "Reorder point": C.num_cfg("Reorder point", "%.1f"), "Stock value": C.money_cfg("Stock value (₹)")})
    C.csv_button(alerts.drop(columns=["status"]).assign(status=alerts["status"].astype(str)), "vyaparai_inventory_alerts.csv", "Download alerts (CSV)", key="inv_dl")
    if len(alerts):
        st.markdown("#### Alert detail")
        chosen = st.selectbox("Product", alerts["product_name"].tolist(), key="inv_detail")
        r = alerts[alerts["product_name"] == chosen].iloc[0]
        dos = "no demand" if not np.isfinite(r["days_of_stock"]) else f"{r['days_of_stock']:.1f}"
        d1, d2 = st.columns([2, 3])
        with d1:
            st.markdown(f"**{r['product_name']}** · {r['category']}  \nStatus: **{r['status']}** · Confidence: **{r['confidence']}**")
            st.markdown(f"- Current stock: **{num(r['current_stock'])}** (+{num(r['incoming_qty'])} on order)\n"
                        f"- Average daily demand: **{r['avg_daily_demand']:.2f}** ({r['demand_source']})\n"
                        f"- Days of stock left: **{dos}**\n"
                        f"- Supplier lead time: **{r['lead_time_days']:.0f} days**\n"
                        f"- Reorder point: **{r['reorder_point']:.1f}** (safety stock {r['safety_stock']:.1f})")
        with d2:
            st.markdown(f"**Recommended action:** {r['action']}")
            st.markdown(f"**Why:** {r['explanation']}")
            if r["data_warnings"]:
                st.warning(r["data_warnings"])
            st.caption("Suggestions only: VyaparAI never changes stock or places orders.")

with tab_all:
    full = inv.copy()
    full["status"] = full["status"].astype(str)
    full["days_of_stock"] = full["days_of_stock"].replace(np.inf, np.nan)
    st.dataframe(full.drop(columns=["product_id", "explanation", "action", "excess_value", "daily_gp_at_risk"]), hide_index=True, width="stretch", column_config={
        "inventory_value": C.money_cfg("Stock value (₹)"), "unit_cost": C.money2_cfg("Unit cost (₹)"), "selling_price": C.money2_cfg("Selling price (₹)"),
        "margin_pct": C.pct_cfg("Margin"), "avg_daily_demand": C.num_cfg("Avg daily demand", "%.2f"), "days_of_stock": C.num_cfg("Days of stock", "%.1f")})
    C.csv_button(full, "vyaparai_inventory_analysis.csv", "Download full analysis (CSV)", key="inv_dl_all")

with tab_promo:
    st.markdown("**Candidates for a discount or promotion**: stock that is slow, dead or in excess and still has a positive margin, "
                "so a modest discount does not mean selling below cost.")
    cand = inv[inv["status"].astype(str).isin(["Slow-moving", "Dead stock", "Excess stock"]) & (inv["margin_pct"] > 0)].copy()
    cand["max_discount_pct"] = cand["margin_pct"]
    cand["status"] = cand["status"].astype(str)
    if cand.empty:
        st.success("No promotion candidates right now.")
    else:
        st.dataframe(cand[["product_name", "status", "current_stock", "inventory_value", "avg_daily_demand", "days_since_last_sale", "margin_pct", "max_discount_pct"]].rename(columns={
            "product_name": "Product", "status": "Status", "current_stock": "Stock", "inventory_value": "Stock value", "avg_daily_demand": "Demand/day",
            "days_since_last_sale": "Days since last sale", "margin_pct": "Margin", "max_discount_pct": "Break-even discount"}), hide_index=True, width="stretch",
            column_config={"Stock value": C.money_cfg("Stock value (₹)"), "Margin": C.pct_cfg("Margin"), "Break-even discount": C.pct_cfg("Break-even discount"),
                           "Demand/day": C.num_cfg("Demand/day", "%.2f")})
        st.caption("Break-even discount = the discount at which the selling price equals unit cost. Any discount below it keeps some gross margin; "
                   "above it every sale loses money. Whether a promotion will actually lift sales is not known from this data.")
    st.markdown("**High-value stock with low sales velocity**")
    hv = inv[inv["status"].astype(str).isin(["Slow-moving", "Dead stock", "Excess stock"])].sort_values("inventory_value", ascending=False).head(10)
    st.dataframe(hv[["product_name", "current_stock", "inventory_value", "avg_daily_demand", "days_of_stock"]].replace(np.inf, np.nan).rename(columns={
        "product_name": "Product", "current_stock": "Stock", "inventory_value": "Stock value", "avg_daily_demand": "Demand/day", "days_of_stock": "Days of stock"}),
        hide_index=True, width="stretch", column_config={"Stock value": C.money_cfg("Stock value (₹)"), "Demand/day": C.num_cfg("Demand/day", "%.2f"),
                                                         "Days of stock": C.num_cfg("Days of stock", "%.0f")})
    st.markdown("**Products with low or negative margin**")
    lm = inv[inv["margin_pct"] < rules.low_margin_pct].sort_values("margin_pct")
    st.dataframe(lm[["product_name", "unit_cost", "selling_price", "margin_pct"]].rename(columns={"product_name": "Product", "unit_cost": "Unit cost", "selling_price": "Selling price", "margin_pct": "Margin"}),
                 hide_index=True, width="stretch", column_config={"Unit cost": C.money2_cfg("Unit cost (₹)"), "Selling price": C.money2_cfg("Selling price (₹)"), "Margin": C.pct_cfg("Margin")})

with tab_how:
    st.markdown(f"""
**Average daily demand** = units sold (excluding returns and duplicate lines) in the last **{rules.demand_window_days} days** ÷ days observed.

**Reorder point** = demand during supplier lead time + safety stock
**Safety stock** = z × standard deviation of daily demand × √lead time (z = **{rules.low_stock_service_level_z}**)

**Suggested order** = order-up-to level − current stock − stock already on order (never below zero), where
order-up-to level = daily demand × (lead time + **{rules.target_coverage_days}** target coverage days) + safety stock.
Purchase lines dated after the latest sale are treated as stock on order.

**Statuses** (first match wins): Stocked out → Reorder now → Stock-out risk (stock alone is shorter than lead time, rescued only by an open order) →
Dead stock (no sale for **{rules.dead_stock_days_since_sale}+** days) → Slow-moving (under **{rules.slow_moving_daily_demand}** units/day or no sale for **{rules.slow_moving_days_since_sale}+** days) →
Excess stock (more than **{rules.excess_stock_days_cover}** days of cover) → Healthy.

**Slow-moving versus dead versus unsellable.** Slow-moving stock still sells, just slowly. Dead stock has not sold for a long time but may be
sellable with a promotion or in the right season. *Unsellable* (expired, damaged, obsolete) cannot be known from sales data, so check the physical stock before writing anything off.

**Confidence.** Low = under {rules.min_history_days} days of history or very few sale days (the file's own reorder level is used instead). Medium = intermittent demand. High = regular sales.
These are estimates from past sales, not guarantees.
""")
    st.markdown("**Adjust thresholds**")
    C.rules_editor(ctx, ["low_stock_service_level_z", "target_coverage_days", "demand_window_days", "slow_moving_daily_demand",
                         "slow_moving_days_since_sale", "dead_stock_days_since_sale", "excess_stock_days_cover", "low_margin_pct", "min_history_days"], "inv_rules")

with tab_rec:
    st.write("Compares the stock recorded in your inventory file with the stock implied by your purchase and sales records (purchases minus net sales). "
             "This works best when the purchase file includes opening stock. Differences can come from duplicates, typos, shrinkage or missing purchase entries.")
    rec = I.reconcile_inventory(ctx.ds)
    if rec.empty:
        st.info("Add a purchases file to enable reconciliation.")
    else:
        diff = rec[rec["difference"].abs() > 0.5]
        st.markdown(f"**{len(rec) - len(diff)}** of {len(rec)} products reconcile exactly; **{len(diff)}** differ.")
        st.dataframe(diff.rename(columns={"product_id": "ID", "product_name": "Product", "recorded_stock": "Recorded stock", "implied_stock": "Implied stock", "difference": "Difference"}),
                     hide_index=True, width="stretch")
