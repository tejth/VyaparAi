"""Reusable Streamlit building blocks shared by all pages."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import config
from src.ui import state, theme
from src.utils.formatting import inr, inr_short, pct


def sidebar_status(ctx: state.Ctx) -> None:
    with st.sidebar:
        st.divider()
        label = "Demo data (synthetic)" if ctx.active == loader_demo() else "Your uploaded data"
        st.markdown(theme.badge(label, theme.C_WARN if ctx.active == loader_demo() else theme.C_ACCENT), unsafe_allow_html=True)
        st.caption(f"Data through {ctx.ds.as_of:%d %b %Y} · {len(ctx.ds.sales):,} sales lines")
        mode, detail = state.llm_mode_summary()
        st.caption(f"{'🟢' if mode.startswith('AI') else '🟠'} {mode}")


def loader_demo() -> str:
    from src.data.loader import DEMO_LABEL
    return DEMO_LABEL


def demo_banner(ctx: state.Ctx) -> None:
    if ctx.active == loader_demo():
        st.info("You are viewing **synthetic demonstration data** for a fictional retailer. Findings are illustrations, not real business results. "
                "Upload your own CSV files in **Data Management** to analyse a real business.", icon="ℹ️")


def date_category_filters(ctx: state.Ctx, key: str, with_categories: bool = True) -> tuple[pd.Timestamp, pd.Timestamp, list[str]]:
    ds = ctx.ds
    c1, c2 = st.columns([2, 3])
    lo, hi = ds.first_date.date(), ds.as_of.date()
    with c1:
        rng = st.date_input("Date range", value=(lo, hi), min_value=lo, max_value=hi, key=f"{key}_dates")
    cats: list[str] = []
    if with_categories:
        with c2:
            all_cats = sorted(ds.sales["category"].dropna().unique().tolist())
            cats = st.multiselect("Categories (empty = all)", all_cats, key=f"{key}_cats")
    if isinstance(rng, (list, tuple)) and len(rng) == 2:
        start, end = pd.Timestamp(rng[0]), pd.Timestamp(rng[1])
    else:  # user is mid-selection
        start, end = pd.Timestamp(lo), pd.Timestamp(hi)
    return start, end, cats


def kpis(items: list[tuple[str, str, str | None]], per_row: int = 5) -> None:
    """items: (label, value, help)."""
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row)
        for col, (label, value, help_) in zip(cols, items[i:i + per_row]):
            col.metric(label, value, help=help_)


def csv_button(df: pd.DataFrame, filename: str, label: str = "Download CSV", key: str | None = None) -> None:
    st.download_button(label, df.to_csv(index=False).encode("utf-8-sig"), file_name=filename, mime="text/csv", key=key or f"dl_{filename}")


def money_cfg(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="₹%.0f")


def money2_cfg(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="₹%.2f")


def pct_cfg(label: str) -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format="%.1f%%")


def num_cfg(label: str, fmt: str = "%.0f") -> st.column_config.NumberColumn:
    return st.column_config.NumberColumn(label, format=fmt)


def filter_inventory(ctx: state.Ctx, cats: list[str]) -> pd.DataFrame:
    inv = ctx.analysis.inventory
    return inv[inv["category"].isin(cats)] if cats else inv


def filter_anomalies(ctx: state.Ctx, start: pd.Timestamp, end: pd.Timestamp, cats: list[str]) -> pd.DataFrame:
    an = ctx.analysis.anomalies
    if an.empty:
        return an
    cat_of = ctx.ds.sales.drop_duplicates("product_id").set_index("product_id")["category"]
    out = an[(pd.to_datetime(an["date"]) >= start) & (pd.to_datetime(an["date"]) <= end)]
    if cats:
        pid_cat = out["product_id"].map(cat_of)
        out = out[pid_cat.isin(cats) | (out["product_id"] == "")]
    return out


def recommendation_card(rec, idx: int) -> None:
    icon = {"High": "🔴", "Medium": "🟠", "Low": "⚪"}[rec.priority]
    exposure = f" · est. exposure {inr_short(rec.exposure_inr)}" if pd.notna(rec.exposure_inr) else ""
    with st.expander(f"{icon} **{rec.priority}** · {rec.title}{exposure}", expanded=False):
        st.markdown(f"**Affected:** {rec.affected}")
        st.markdown("**Supporting evidence**\n" + "\n".join(f"- {e}" for e in rec.evidence))
        st.markdown(f"**Recommended action:** {rec.action}")
        if pd.notna(rec.exposure_inr):
            note = "" if rec.counted else " _(overlaps a higher-priority item for the same product, so it is not added to the total)_"
            st.markdown(f"**Estimated financial exposure:** {inr(rec.exposure_inr)}{note}  \n_Basis: {rec.exposure_basis}_")
        else:
            st.markdown("**Estimated financial exposure:** not calculable from the available data.")
        if rec.assumptions:
            st.caption(f"Assumptions and limitations: {rec.assumptions}")


def status_caption(ctx: state.Ctx) -> None:
    s = ctx.analysis.summary
    parts = [f"Gross profit is revenue minus cost of goods sold; it is **not** net profit (overheads such as rent and salaries are not in the data)."]
    if s.duplicates_excluded:
        parts.append(f"{s.duplicates_excluded} duplicate line(s) are excluded.")
    if pd.notna(s.cost_coverage_pct) and s.cost_coverage_pct < 99.95:
        parts.append(f"Cost is known for {pct(s.cost_coverage_pct)} of revenue; the rest is excluded from margin figures.")
    st.caption(" ".join(parts))


# --------------------------------------------------------------------------- configurable thresholds
RULE_META: dict[str, tuple[str, str, float, float, float]] = {
    "low_stock_service_level_z": ("Safety-stock z-score", "1.28 = 90%, 1.65 = 95%, 2.33 = 99% service level. Higher means more safety stock.", 0.0, 4.0, 0.05),
    "target_coverage_days": ("Target coverage (days)", "A reorder tops stock up to cover lead time plus this many days of demand.", 1, 180, 1),
    "demand_window_days": ("Demand window (days)", "Recent period used to estimate average daily demand.", 14, 180, 1),
    "slow_moving_daily_demand": ("Slow-moving: units/day below", "Products selling fewer units per day than this are slow-moving.", 0.0, 5.0, 0.05),
    "slow_moving_days_since_sale": ("Slow-moving: days since last sale", "Also slow-moving if nothing sold for this many days.", 7, 365, 1),
    "dead_stock_days_since_sale": ("Dead stock: days since last sale", "Stock with no sale for this long is flagged as dead stock (review, not necessarily unsellable).", 30, 730, 1),
    "excess_stock_days_cover": ("Excess stock: days of cover above", "Stock covering more days of demand than this is excess.", 30, 730, 1),
    "low_margin_pct": ("Low margin below (%)", "Gross margin below this is flagged as low margin.", 0.0, 50.0, 0.5),
    "high_discount_pct": ("Large discount from (%)", "Discounts at or above this are flagged.", 5.0, 90.0, 1.0),
    "large_qty_zscore": ("Large quantity sensitivity", "Higher = fewer 'abnormally large quantity' flags.", 1.0, 10.0, 0.5),
    "price_change_pct": ("Price deviation from usual (%)", "Unit prices differing from the product's usual price by more than this are flagged.", 5.0, 200.0, 1.0),
    "spike_zscore": ("Daily revenue spike sensitivity", "Higher = fewer revenue-spike flags.", 1.0, 10.0, 0.5),
    "decline_pct": ("Sales decline threshold (%)", "A product is 'declining' if units fall at least this much versus the previous period.", 5.0, 90.0, 1.0),
    "min_history_days": ("Minimum history (days)", "Below this much history, demand estimates are marked low confidence.", 7, 120, 1),
}


def rules_editor(ctx: state.Ctx, fields: list[str], key: str) -> None:
    """Edit and persist a subset of business-rule thresholds. Saving recomputes all analyses."""
    import dataclasses

    vals: dict[str, float] = {}
    cols = st.columns(3)
    for i, name in enumerate(fields):
        label, help_, lo, hi, step = RULE_META[name]
        current = getattr(ctx.rules, name)
        is_int = isinstance(getattr(config.DEFAULT_RULES, name), int)
        with cols[i % 3]:
            vals[name] = st.number_input(label, min_value=type(current)(lo), max_value=type(current)(hi), value=current,
                                         step=int(step) if is_int else float(step), help=help_, key=f"{key}_{name}")
    b1, b2, _ = st.columns([1, 1, 3])
    if b1.button("Save thresholds", key=f"{key}_save", type="primary"):
        saved = ctx.repo.get_config(state.RULES_KEY, {}) or {}
        saved.update({k: v for k, v in vals.items()})
        ctx.repo.set_config(state.RULES_KEY, saved)
        st.rerun()
    if b2.button("Reset to defaults", key=f"{key}_reset"):
        saved = ctx.repo.get_config(state.RULES_KEY, {}) or {}
        for name in fields:
            saved.pop(name, None)
        ctx.repo.set_config(state.RULES_KEY, saved)
        st.rerun()
    changed = [n for n in fields if getattr(ctx.rules, n) != getattr(config.DEFAULT_RULES, n)]
    if changed:
        st.caption("Customised thresholds: " + ", ".join(RULE_META[n][0] for n in changed))
    _ = dataclasses
