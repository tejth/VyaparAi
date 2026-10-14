"""Read-only business-analysis tools used by the AI analyst.

Security model
--------------
* The model can only *name* a tool from the fixed registry below and supply arguments. Arguments are validated against a
  declared schema (type, range, allowed values); unknown tools / unknown arguments are rejected.
* Tools read pre-computed, in-memory analytics for the active dataset. They never accept SQL, Python, shell commands or file
  paths, never write to the database or inventory, and never place orders.
* Every result states the period covered and whether numbers are observed facts or estimates.
"""
from __future__ import annotations

import difflib
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from src.analytics import inventory as I
from src.analytics import profitability as P
from src.analytics import sales as S
from src.analytics.pipeline import Analysis
from src.analytics import recommendations as R
from src.ml import forecasting as F
from src.utils.dates import period_label
from src.utils.formatting import inr, num, pct


class ToolArgumentError(ValueError):
    """Raised when a tool name or its arguments fail validation."""


@dataclass
class ToolResult:
    tool: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""


# --------------------------------------------------------------------------- tool specs (the whole allowed surface)
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "get_business_summary": {
        "description": "Revenue, cost of goods sold, gross profit, margin, units and comparison with the previous equal period.",
        "params": {"days": {"type": int, "min": 7, "max": 365, "default": 30}}},
    "get_loss_making_products": {
        "description": "Products whose gross profit is negative over the period (selling below cost).",
        "params": {"days": {"type": int, "min": 7, "max": 365, "default": 90}, "top_n": {"type": int, "min": 1, "max": 25, "default": 10}}},
    "get_reorder_recommendations": {
        "description": "Products that need reordering or are out of stock, with suggested order quantities (not placed automatically).",
        "params": {"top_n": {"type": int, "min": 1, "max": 25, "default": 10}}},
    "get_inventory_risk": {
        "description": "Products with the highest inventory risk: stock-outs, reorder needs, dead, slow-moving and excess stock.",
        "params": {"top_n": {"type": int, "min": 1, "max": 25, "default": 10}}},
    "get_category_performance": {
        "description": "Revenue, gross profit and margin by category for a period.",
        "params": {"days": {"type": int, "min": 7, "max": 365, "default": 30}}},
    "get_profit_leaks": {
        "description": "Biggest potential profit leaks: negative margins, heavy discounts, duplicates, inconsistent totals and stock problems.",
        "params": {"top_n": {"type": int, "min": 1, "max": 15, "default": 8}}},
    "get_declining_products": {
        "description": "Products selling fewer units than in the previous equal period.",
        "params": {"days": {"type": int, "min": 14, "max": 90, "default": 30}, "top_n": {"type": int, "min": 1, "max": 25, "default": 10}}},
    "explain_profit_change": {
        "description": "Why gross profit changed versus the previous equal period (volume vs margin effects, top contributors).",
        "params": {"days": {"type": int, "min": 14, "max": 180, "default": 30}}},
    "get_demand_forecast": {
        "description": "Forecast of daily demand for one named product for the next 7 or 30 days, with validation accuracy.",
        "params": {"product": {"type": str, "max_len": 80}, "horizon": {"type": int, "choices": [7, 30], "default": 7}}},
    "get_improvement_actions": {
        "description": "The top ranked actions to improve inventory efficiency or profit, with evidence.",
        "params": {"top_n": {"type": int, "min": 1, "max": 10, "default": 5}, "focus": {"type": str, "choices": ["inventory", "profit", "all"], "default": "all"}}},
}


def validate_args(tool: str, args: dict[str, Any] | None) -> dict[str, Any]:
    """Validate a tool call. Returns cleaned arguments or raises ToolArgumentError."""
    if tool not in TOOL_SPECS:
        raise ToolArgumentError(f"Unknown tool '{tool}'. Allowed tools: {', '.join(TOOL_SPECS)}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolArgumentError("Tool arguments must be an object")
    spec = TOOL_SPECS[tool]["params"]
    unknown = set(args) - set(spec)
    if unknown:
        raise ToolArgumentError(f"Unknown argument(s) for {tool}: {', '.join(sorted(unknown))}")
    clean: dict[str, Any] = {}
    for name, rule in spec.items():
        if name not in args or args[name] is None:
            if "default" in rule:
                clean[name] = rule["default"]
                continue
            raise ToolArgumentError(f"Missing required argument '{name}' for {tool}")
        v = args[name]
        if rule["type"] is int:
            if isinstance(v, bool):
                raise ToolArgumentError(f"'{name}' must be an integer")
            try:
                fv = float(v)
            except (TypeError, ValueError) as e:
                raise ToolArgumentError(f"'{name}' must be an integer") from e
            if not math.isfinite(fv) or fv != int(fv):
                raise ToolArgumentError(f"'{name}' must be a whole number")
            v = int(fv)
            if "choices" in rule and v not in rule["choices"]:
                raise ToolArgumentError(f"'{name}' must be one of {rule['choices']}")
            if "min" in rule and not (rule["min"] <= v <= rule["max"]):
                raise ToolArgumentError(f"'{name}' must be between {rule['min']} and {rule['max']}")
        elif rule["type"] is str:
            if not isinstance(v, str):
                raise ToolArgumentError(f"'{name}' must be text")
            v = v.strip()
            if "choices" in rule and v not in rule["choices"]:
                raise ToolArgumentError(f"'{name}' must be one of {rule['choices']}")
            if len(v) == 0 or len(v) > rule.get("max_len", 80):
                raise ToolArgumentError(f"'{name}' must be 1-{rule.get('max_len', 80)} characters")
        clean[name] = v
    return clean


def tool_catalog() -> str:
    lines = []
    for name, spec in TOOL_SPECS.items():
        params = ", ".join(f"{p}" + (f" (default {r['default']})" if "default" in r else " (required)") for p, r in spec["params"].items())
        lines.append(f"- {name}({params}): {spec['description']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- implementations
def _r(x: Any, nd: int = 2) -> Any:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    if isinstance(x, (np.floating, float)):
        return round(float(x), nd)
    if isinstance(x, (np.integer,)):
        return int(x)
    return x


class BusinessTools:
    """Executes validated tool calls against an in-memory ``Analysis``."""

    def __init__(self, analysis: Analysis) -> None:
        self.a = analysis
        self.ds = analysis.ds
        self._impl: dict[str, Callable[..., dict[str, Any]]] = {
            "get_business_summary": self.get_business_summary, "get_loss_making_products": self.get_loss_making_products,
            "get_reorder_recommendations": self.get_reorder_recommendations, "get_inventory_risk": self.get_inventory_risk,
            "get_category_performance": self.get_category_performance, "get_profit_leaks": self.get_profit_leaks,
            "get_declining_products": self.get_declining_products, "explain_profit_change": self.explain_profit_change,
            "get_demand_forecast": self.get_demand_forecast, "get_improvement_actions": self.get_improvement_actions,
        }

    # ---- dispatch
    def call(self, tool: str, args: dict[str, Any] | None = None) -> ToolResult:
        try:
            clean = validate_args(tool, args)
        except ToolArgumentError as e:
            return ToolResult(tool, False, error=str(e))
        if self.ds.is_empty:
            return ToolResult(tool, False, error="No sales data is loaded, so this question cannot be answered.")
        try:
            return ToolResult(tool, True, self._impl[tool](**clean))
        except Exception as e:  # tools must never crash the chat
            return ToolResult(tool, False, error=f"The calculation failed: {type(e).__name__}")

    def _window(self, days: int) -> tuple[pd.Timestamp, pd.Timestamp]:
        end = self.ds.as_of
        return max(end - pd.Timedelta(days=days - 1), self.ds.first_date), end

    # ---- tools
    def get_business_summary(self, days: int) -> dict[str, Any]:
        start, end = self._window(days)
        cur = P.compute_summary(P.filter_sales(self.ds.sales, start, end))
        out: dict[str, Any] = {"period": period_label(start, end), "type": "observed",
                               "revenue": inr(cur.revenue), "cogs": inr(cur.cogs), "gross_profit": inr(cur.gross_profit),
                               "gross_margin": pct(cur.gross_margin_pct), "units_sold": num(cur.units_sold),
                               "transactions": cur.transactions, "active_products": cur.active_products,
                               "discounts_given": inr(cur.discount_value), "refunds": inr(cur.refund_value),
                               "cost_coverage": pct(cur.cost_coverage_pct),
                               "notes": ["Gross profit = revenue - cost of goods sold. It is not net profit: rent, salaries and other overheads are not in the data."]}
        cmp = S.period_comparison(self.ds.sales, self.ds.as_of, days)
        if cmp.sufficient:
            out["previous_period"] = cmp.previous_label
            out["comparison"] = [{"metric": m.metric, "current": _r(m.current), "previous": _r(m.previous), "change_pct": _r(m.change_pct, 1)}
                                 for m in cmp.metrics.itertuples()]
        else:
            out["notes"].append(cmp.note)
        return out

    def get_loss_making_products(self, days: int, top_n: int) -> dict[str, Any]:
        start, end = self._window(days)
        prod = P.by_product(P.filter_sales(self.ds.sales, start, end))
        loss = P.loss_making_products(prod).head(top_n)
        return {"period": period_label(start, end), "type": "observed",
                "count_loss_making": int(len(P.loss_making_products(prod))),
                "products": [{"product": r.product_name, "revenue": inr(r.revenue), "gross_profit": inr(r.gross_profit),
                              "margin": pct(r.margin_pct), "units": num(r.units)} for r in loss.itertuples()],
                "notes": ["Gross loss uses recorded unit costs; discounts and cost increases both reduce margin."] if len(loss) else
                         ["No product has a negative gross profit in this period."]}

    def get_reorder_recommendations(self, top_n: int) -> dict[str, Any]:
        inv = self.a.inventory
        need = inv[inv["status"].astype(str).isin(["Stocked out", "Reorder now", "Stock-out risk"])].sort_values(
            ["status", "daily_gp_at_risk"], ascending=[True, False]).head(top_n)
        return {"as_of": f"{self.ds.as_of:%d %b %Y}", "type": "recommendation",
                "products": [{"product": r.product_name, "status": str(r.status), "current_stock": num(r.current_stock),
                              "on_order": num(r.incoming_qty), "avg_daily_demand": _r(r.avg_daily_demand),
                              "days_of_stock": _r(r.days_of_stock, 1) if np.isfinite(r.days_of_stock) else None,
                              "lead_time_days": num(r.lead_time_days), "suggested_order_qty": int(r.suggested_order_qty),
                              "reason": r.explanation, "confidence": r.confidence, "data_warnings": r.data_warnings} for r in need.itertuples()],
                "total_needing_action": int(inv["status"].astype(str).isin(["Stocked out", "Reorder now", "Stock-out risk"]).sum()),
                "notes": ["Quantities are suggestions from recent sales; nothing is ordered automatically. Review with your supplier terms and cash position."]}

    def get_inventory_risk(self, top_n: int) -> dict[str, Any]:
        inv = self.a.inventory
        risky = inv[~inv["status"].astype(str).isin(["Healthy", "No demand data"])]
        sc = I.summary_counts(inv)
        return {"as_of": f"{self.ds.as_of:%d %b %Y}", "type": "observed",
                "counts": {"stocked_out": sc["stocked_out"], "needs_reorder_or_at_risk": sc["low_stock"], "slow_or_dead": sc["slow_moving"], "dead_stock": sc["dead_stock"]},
                "inventory_value_at_cost": inr(sc["inventory_value"]),
                "products": [{"product": r.product_name, "status": str(r.status), "stock": num(r.current_stock), "value_at_cost": inr(r.inventory_value),
                              "avg_daily_demand": _r(r.avg_daily_demand), "days_since_last_sale": r.days_since_last_sale, "action": r.action}
                             for r in risky.head(top_n).itertuples()]}

    def get_category_performance(self, days: int) -> dict[str, Any]:
        start, end = self._window(days)
        cat = P.by_category(P.filter_sales(self.ds.sales, start, end))
        return {"period": period_label(start, end), "type": "observed",
                "categories": [{"category": r.category, "revenue": inr(r.revenue), "gross_profit": inr(r.gross_profit), "margin": pct(r.margin_pct),
                                "revenue_share": pct(r.revenue_share_pct)} for r in cat.itertuples()],
                "best_by_revenue": cat.iloc[0]["category"] if len(cat) else None,
                "best_by_margin": cat.sort_values("margin_pct", ascending=False).iloc[0]["category"] if len(cat) else None}

    def get_profit_leaks(self, top_n: int) -> dict[str, Any]:
        recs = [r for r in self.a.recommendations if r.kind in {"negative_margin", "discounting", "large_discounts", "duplicates", "totals", "slow_stock", "stockout"}]
        recs = [r for r in recs if not pd.isna(r.exposure_inr)]
        recs.sort(key=lambda r: -r.exposure_inr)
        return {"period": f"{self.ds.first_date:%d %b %Y} to {self.ds.as_of:%d %b %Y}", "type": "estimate",
                "leaks": [{"title": r.title, "estimated_exposure": inr(r.exposure_inr), "basis": r.exposure_basis, "evidence": r.evidence,
                           "counted_in_total": r.counted} for r in recs[:top_n]],
                "total_exposure_counted_once": inr(R.total_counted_exposure(self.a.recommendations)),
                "notes": ["Exposure figures are estimates with stated bases, not guaranteed savings. Overlapping items are counted once in the total."]}

    def get_declining_products(self, days: int, top_n: int) -> dict[str, Any]:
        dec = S.declining_products(self.ds.sales, self.ds.as_of, days, 30.0)
        end = self.ds.as_of
        cur_start = end - pd.Timedelta(days=days - 1)
        return {"period": period_label(cur_start, end), "previous_period": period_label(cur_start - pd.Timedelta(days=days), cur_start - pd.Timedelta(days=1)),
                "type": "observed", "count": int(len(dec)),
                "products": [{"product": r.product_name, "units_previous": num(r.units_prev), "units_current": num(r.units_cur), "change": pct(r.change_pct, 0)}
                             for r in dec.head(top_n).itertuples()],
                "notes": ["A drop in units can come from stock-outs, seasonality or price changes; the data alone does not prove the cause."]}

    def explain_profit_change(self, days: int) -> dict[str, Any]:
        end = self.ds.as_of
        cs = end - pd.Timedelta(days=days - 1)
        pe = cs - pd.Timedelta(days=1)
        ps = pe - pd.Timedelta(days=days - 1)
        if self.ds.first_date > ps:
            return {"type": "observed", "notes": [f"Not enough history to compare two {days}-day periods."]}
        cur = P.by_product(P.filter_sales(self.ds.sales, cs, end)).set_index("product_id")
        prev = P.by_product(P.filter_sales(self.ds.sales, ps, pe)).set_index("product_id")
        m = prev[["product_name", "units", "gross_profit", "revenue", "cogs", "avg_discount_pct"]].join(
            cur[["units", "gross_profit", "revenue", "cogs", "avg_discount_pct"]], how="outer", lsuffix="_p", rsuffix="_c")
        m["product_name"] = m["product_name"].fillna(cur["product_name"])
        for c in ("units_p", "units_c", "gross_profit_p", "gross_profit_c", "revenue_p", "revenue_c"):
            m[c] = m[c].fillna(0.0)
        gpu_p = (m["gross_profit_p"] / m["units_p"].replace(0, np.nan))
        gpu_c = (m["gross_profit_c"] / m["units_c"].replace(0, np.nan))
        vol = (m["units_c"] - m["units_p"]) * gpu_p.fillna(gpu_c).fillna(0.0)
        rate = (m["units_c"] * (gpu_c - gpu_p)).fillna(0.0)
        m["volume_effect"], m["margin_effect"] = vol, rate
        m["change"] = m["gross_profit_c"] - m["gross_profit_p"]
        residual = m["change"] - vol - rate
        total_change = float(m["change"].sum())
        worst = m.sort_values("change").head(5)
        return {"period": period_label(cs, end), "previous_period": period_label(ps, pe), "type": "observed",
                "gross_profit_current": inr(m["gross_profit_c"].sum()), "gross_profit_previous": inr(m["gross_profit_p"].sum()),
                "change": inr(total_change), "direction": "decreased" if total_change < 0 else "increased",
                "volume_effect": inr(vol.sum()), "margin_effect": inr(rate.sum()), "unexplained_residual": inr(residual.sum()),
                "top_negative_contributors": [{"product": r.product_name, "change": inr(r.change), "volume_effect": inr(r.volume_effect),
                                               "margin_effect": inr(r.margin_effect),
                                               "discount_current": pct(r.avg_discount_pct_c), "discount_previous": pct(r.avg_discount_pct_p)} for r in worst.itertuples()],
                "notes": ["Volume effect = change in units x previous gross profit per unit. Margin effect = current units x change in gross profit per unit "
                          "(price, discount or cost). These describe what changed, not why; seasonality and stock-outs also matter."]}

    def resolve_product(self, text: str) -> tuple[str | None, list[str]]:
        """Match free text to a product id. Returns (product_id, []) or (None, suggestions)."""
        names = self.ds.sales.drop_duplicates("product_id").set_index("product_id")["product_name"].astype(str)
        low = text.lower().strip()
        lower_names = names.str.lower()
        exact = names.index[(lower_names == low) | (names.index.str.lower() == low)]
        if len(exact):
            return str(exact[0]), []
        contains = names.index[lower_names.str.contains(low, regex=False)]
        if len(contains) == 1:
            return str(contains[0]), []
        if len(contains) > 1:
            return None, [str(n) for n in names[contains].head(5)]
        tokens = [t for t in low.replace("-", " ").split() if len(t) > 1]
        if tokens:
            tok_hit = names.index[lower_names.apply(lambda n: all(t in n for t in tokens))]
            if len(tok_hit) == 1:
                return str(tok_hit[0]), []
            if len(tok_hit) > 1:
                return None, [str(n) for n in names[tok_hit].head(5)]
        close = difflib.get_close_matches(low, list(lower_names), n=3, cutoff=0.5)
        suggestions = [str(names[lower_names == c].iloc[0]) for c in close]
        if close and difflib.SequenceMatcher(None, low, close[0]).ratio() > 0.8:
            return str(names.index[lower_names == close[0]][0]), []
        return None, suggestions

    def get_demand_forecast(self, product: str, horizon: int) -> dict[str, Any]:
        pid, suggestions = self.resolve_product(product)
        if pid is None:
            return {"type": "estimate", "found": False, "notes": [f"No single product matches '{product}'." + (f" Did you mean: {', '.join(suggestions)}?" if suggestions else "")]}
        res = F.forecast_product(self.ds.sales, pid, horizon)
        out: dict[str, Any] = {"type": "estimate", "found": True, "product": res.product_name, "horizon_days": horizon, "warnings": res.warnings,
                               "sufficient_data": res.sufficient}
        if res.sufficient:
            out.update(model_used=F.MODEL_LABELS.get(res.selected_model, res.selected_model), expected_total_units=_r(res.expected_units, 1),
                       avg_daily_units=_r(res.avg_daily_forecast, 2), forecast_from=f"{res.forecast['date'].min():%d %b %Y}",
                       forecast_to=f"{res.forecast['date'].max():%d %b %Y}",
                       validation=[{"model": F.MODEL_LABELS.get(r.model, r.model), "MAE": _r(r.mae, 3), "WAPE": pct(r.wape * 100) if pd.notna(r.wape) else "n/a"}
                                   for r in res.metrics.itertuples()],
                       notes=["Forecasts are estimates, not certainty; the range comes from recent validation errors."])
            if pd.notna(res.forecast["lower"].iloc[0]):
                out["approx_range_total_units"] = [_r(res.forecast["lower"].sum(), 1), _r(res.forecast["upper"].sum(), 1)]
        return out

    def get_improvement_actions(self, top_n: int, focus: str) -> dict[str, Any]:
        kinds = {"inventory": {"stockout", "reorder", "slow_stock"}, "profit": {"negative_margin", "discounting", "large_discounts", "duplicates", "totals"}}
        recs = [r for r in self.a.recommendations if focus == "all" or r.kind in kinds[focus]]
        return {"type": "recommendation", "focus": focus, "as_of": f"{self.ds.as_of:%d %b %Y}",
                "data_period": f"{self.ds.first_date:%d %b %Y} to {self.ds.as_of:%d %b %Y}",
                "actions": [{"priority": r.priority, "title": r.title, "evidence": r.evidence, "action": r.action,
                             "estimated_exposure": None if pd.isna(r.exposure_inr) else inr(r.exposure_inr), "basis": r.exposure_basis,
                             "assumptions": r.assumptions} for r in recs[:top_n]],
                "notes": ["Actions are ranked by priority then estimated exposure. Exposure figures are estimates, not promised savings."]}
