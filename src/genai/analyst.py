"""The AI Business Analyst: deterministic tools first, language model second.

Flow for every question
1. Route the question to a few read-only tools (keyword intent rules; an LLM "planner" is used only if no rule matches
   and a model is available). Tool arguments are validated; invalid calls are dropped.
2. Run the tools in Python. Their structured output (FACTS) is the only source of numbers.
3. If a local model is available, it phrases the answer from FACTS. A numeric grounding check flags figures in the answer
   that do not appear in FACTS. If the model is unavailable or fails, a clearly labelled FALLBACK answer is generated
   from templates.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from src.analytics.pipeline import Analysis
from src.genai import prompts
from src.genai.business_tools import BusinessTools, ToolArgumentError, validate_args
from src.genai.ollama_client import LLMClient, LLMUnavailable
from src.utils.logging_config import get_logger

log = get_logger(__name__)

MAX_QUESTION_LEN = 500
MAX_TOOLS = 3

HELP_TEXT = (
    "I can answer questions about the data you loaded. For example:\n"
    "- Which products are losing me money?\n- What should I reorder this week?\n- Why did gross profit decrease?\n"
    "- Which products have the highest inventory risk?\n- Which categories are performing best?\n"
    "- What are my biggest potential profit leaks?\n- Which products are selling more slowly than before?\n"
    "- Summarize my business performance for the last 30 days.\n- Recommend five actions to improve inventory efficiency.\n"
    "- Forecast demand for <product name> for the next 7 days.")


@dataclass
class AnalystAnswer:
    text: str
    mode: str                                  # "llm" | "fallback" | "limitation"
    backend: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    facts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ungrounded_numbers: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- routing
_WORD_NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}


def sanitize(question: str) -> str:
    q = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", question or "")
    return re.sub(r"\s+", " ", q).strip()[:MAX_QUESTION_LEN]


def _days(q: str, default: int) -> int:
    m = re.search(r"(?:last|past|previous)?\s*(\d{1,3})\s*days?", q)
    if m:
        return max(7, min(int(m.group(1)), 365))
    if re.search(r"\b(this|last|past) week\b|\bweekly\b|\b7 ?d", q):
        return 7
    if re.search(r"\b(this|last|past) month\b|\bmonthly\b", q):
        return 30
    if re.search(r"\bquarter\b|\b3 months\b|\bthree months\b", q):
        return 90
    if re.search(r"\b(this|last|past) year\b|\b12 months\b", q):
        return 365
    return default


def _count(q: str, default: int) -> int:
    m = re.search(r"\b(?:top|best|worst)?\s*(\d{1,2})\s+(?:actions?|products?|items?|recommendations?|things)", q)
    if m:
        return max(1, min(int(m.group(1)), 10))
    for w, n in _WORD_NUM.items():
        if re.search(rf"\b{w}\s+(?:actions?|products?|items?|recommendations?|things|ways)", q):
            return n
    return default


def route(question: str, tools: BusinessTools) -> list[tuple[str, dict[str, Any]]]:
    """Map a question to up to MAX_TOOLS validated tool calls (deterministic keyword rules)."""
    q = question.lower()
    calls: list[tuple[str, dict[str, Any]]] = []

    def add(name: str, args: dict[str, Any] | None = None) -> None:
        if len(calls) < MAX_TOOLS and name not in [c[0] for c in calls]:
            calls.append((name, args or {}))

    if re.search(r"forecast|predict|projection|next (7|30|seven|thirty|week|month)|expected demand|demand for", q):
        pid = _product_in(q, tools)
        if pid:
            add("get_demand_forecast", {"product": pid, "horizon": 30 if re.search(r"30|thirty|month", q) else 7})
        else:
            add("get_demand_forecast", {"product": _extract_product_phrase(q), "horizon": 30 if re.search(r"30|thirty|month", q) else 7})
    if re.search(r"why.*(profit|margin|earn).*(decreas|drop|fall|fell|down|lower|declin|less|reduc)|(profit|margin).*(decreas|drop|fall|fell|declin|lower)", q):
        add("explain_profit_change", {"days": _days(q, 30)})
        add("get_business_summary", {"days": _days(q, 30)})
    if re.search(r"losing (me )?money|loss.?making|unprofitable|negative margin|below cost|selling at a loss|which products.*loss", q):
        add("get_loss_making_products", {"days": _days(q, 90)})
    if re.search(r"reorder|re-order|restock|stock up|what (should|do) i (order|buy)|running out|run out|out of stock|stock-?out|low stock", q):
        add("get_reorder_recommendations", {"top_n": 10})
    if re.search(r"inventory risk|stock risk|dead stock|dead inventory|slow.?moving (stock|inventory|products)|excess (stock|inventory)|unsold|capital (tied|locked)", q):
        add("get_inventory_risk", {"top_n": 10})
    if re.search(r"categor", q):
        add("get_category_performance", {"days": _days(q, 30)})
    if re.search(r"leak|waste|where am i losing|discount|duplicate", q):
        add("get_profit_leaks", {"top_n": 8})
    if re.search(r"slower|slowly|declin|falling|dropping|selling less|sales (are )?down|decreas.*sales", q) and "profit" not in q:
        add("get_declining_products", {"days": min(_days(q, 30), 90), "top_n": 10})
    if re.search(r"(five|5|three|3|top|best|some)\s+(actions?|recommendations?|things|ways|steps)|recommend|improve|how can i|what should i do|suggest|advice|efficien", q):
        focus = "inventory" if re.search(r"inventory|stock", q) else ("profit" if re.search(r"profit|margin", q) else "all")
        add("get_improvement_actions", {"top_n": _count(q, 5), "focus": focus})
    explicit = re.search(r"summar|overview|performance|how (is|am|are|was|were).*(business|doing|going|sales|shop|store)", q)
    generic = re.search(r"revenue|\bsales\b|profit|margin|earn", q)
    if explicit or (generic and not calls):
        add("get_business_summary", {"days": _days(q, 30)})
    return calls


def _product_in(q: str, tools: BusinessTools) -> str | None:
    names = tools.ds.sales.drop_duplicates("product_id")[["product_id", "product_name"]]
    best, best_len = None, 0
    for r in names.itertuples():
        n = str(r.product_name).lower()
        if n in q and len(n) > best_len:
            best, best_len = str(r.product_name), len(n)
    return best


def _extract_product_phrase(q: str) -> str:
    m = re.search(r"(?:for|of)\s+(.+?)(?:\s+(?:for|over|in)\s+the\b|\s+next\b|\?|$)", q)
    return (m.group(1) if m else q).strip(" ?.")[:80] or "product"


# --------------------------------------------------------------------------- grounding check
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> list[float]:
    out = []
    for tok in _NUM.findall(text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            pass
    return out


def ungrounded_numbers(answer: str, facts_text: str, question: str = "") -> list[str]:
    """Numbers in the answer that cannot be traced to FACTS or the question.

    Accepted: an exact match (0.1% tolerance), or a thousand / lakh / crore rescaling of a fact (for example 1.23 for
    ₹1,23,456 written as "1.23 lakh"). Small whole numbers (counts, ranks, days) and years are ignored.
    """
    base = _numbers(facts_text) + _numbers(question)
    scaled = [a / d for a in base for d in (1e3, 1e5, 1e7) if abs(a) >= d]
    bad: list[str] = []
    for tok in _NUM.findall(answer):
        v = float(tok.replace(",", ""))
        if v < 32 and v == int(v):
            continue
        if 2000 <= v <= 2100 and v == int(v):
            continue
        direct = any(abs(v - a) <= max(0.001 * abs(a), 0.51) for a in base)
        rescaled = any(abs(v - a) <= max(0.006 * abs(a), 0.051) for a in scaled)
        if not (direct or rescaled):
            bad.append(tok)
    return list(dict.fromkeys(bad))


# --------------------------------------------------------------------------- fallback rendering
def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items)


def render_fallback(results: list[dict[str, Any]]) -> str:
    """Template answer built only from tool output (used when no language model is available)."""
    parts: list[str] = []
    for item in results:
        t, d = item["tool"], item["result"]
        notes = d.get("notes", [])
        if t == "get_business_summary":
            lines = [f"Gross profit {d['gross_profit']} on revenue {d['revenue']} (gross margin {d['gross_margin']}); cost of goods sold {d['cogs']}.",
                     f"{d['units_sold']} units across {d['transactions']} transactions and {d['active_products']} active products; discounts given {d['discounts_given']}."]
            for c in d.get("comparison", [])[:2]:
                lines.append(f"{c['metric']}: {c['current']:,.0f} vs {c['previous']:,.0f} before ({c['change_pct']:+.1f}%)." if c["change_pct"] is not None else f"{c['metric']}: {c['current']}")
            parts.append(f"**Business summary, {d['period']}** (observed)\n{_bullets(lines)}")
        elif t == "get_loss_making_products":
            rows = [f"{p['product']}: gross profit {p['gross_profit']} on revenue {p['revenue']} (margin {p['margin']})" for p in d["products"]]
            parts.append(f"**Products with negative gross profit, {d['period']}** (observed)\n" + (_bullets(rows) if rows else "None found."))
        elif t == "get_reorder_recommendations":
            rows = [f"{p['product']} ({p['status']}): stock {p['current_stock']}, demand {p['avg_daily_demand']}/day, lead time {p['lead_time_days']} days. "
                    f"Suggested order about {p['suggested_order_qty']} units." for p in d["products"]]
            parts.append(f"**Reorder suggestions as of {d['as_of']}** (recommendation, {d['total_needing_action']} products need action)\n" + (_bullets(rows) if rows else "No reorders needed."))
        elif t == "get_inventory_risk":
            c = d["counts"]
            rows = [f"{p['product']} ({p['status']}): {p['stock']} units worth {p['value_at_cost']}. {p['action']}" for p in d["products"]]
            parts.append(f"**Inventory risk as of {d['as_of']}** (observed): {c['stocked_out']} stocked out, {c['needs_reorder_or_at_risk']} need reorder, "
                         f"{c['slow_or_dead']} slow or dead; stock value at cost {d['inventory_value_at_cost']}.\n{_bullets(rows)}")
        elif t == "get_category_performance":
            rows = [f"{c['category']}: revenue {c['revenue']}, gross profit {c['gross_profit']}, margin {c['margin']}" for c in d["categories"]]
            parts.append(f"**Category performance, {d['period']}** (observed). Highest revenue: {d['best_by_revenue']}; highest margin: {d['best_by_margin']}.\n{_bullets(rows)}")
        elif t == "get_profit_leaks":
            rows = [f"{l['title']}: estimated exposure {l['estimated_exposure']}" + ("" if l["counted_in_total"] else " (overlaps another item, not added to total)") for l in d["leaks"]]
            parts.append(f"**Potential profit leaks, {d['period']}** (estimates). Total counted once: {d['total_exposure_counted_once']}.\n{_bullets(rows)}")
        elif t == "get_declining_products":
            rows = [f"{p['product']}: {p['units_previous']} to {p['units_current']} units ({p['change']})" for p in d["products"]]
            parts.append(f"**Products selling less, {d['period']} vs {d['previous_period']}** (observed, {d['count']} products)\n" + (_bullets(rows) if rows else "None."))
        elif t == "explain_profit_change":
            if "change" not in d:
                parts.append(f"**Gross profit change**\n{_bullets(notes)}")
                continue
            rows = [f"{p['product']}: {p['change']} (volume {p['volume_effect']}, margin {p['margin_effect']})" for p in d["top_negative_contributors"]]
            parts.append(f"**Gross profit {d['direction']} by {d['change'].lstrip('-')}, {d['period']} vs {d['previous_period']}** (observed): {d['gross_profit_previous']} to {d['gross_profit_current']}. "
                         f"Change in units sold accounts for {d['volume_effect']}; change in gross profit per unit (price, discount, cost) accounts for {d['margin_effect']}.\nBiggest negative contributors:\n{_bullets(rows)}")
        elif t == "get_demand_forecast":
            if not d.get("found"):
                parts.append("**Forecast**\n" + _bullets(notes))
            elif not d.get("sufficient_data"):
                parts.append(f"**Forecast for {d['product']}**\n{_bullets(d.get('warnings', []))}")
            else:
                parts.append(f"**Forecast for {d['product']}, {d['forecast_from']} to {d['forecast_to']}** (estimate): about {d['expected_total_units']} units "
                             f"({d['avg_daily_units']}/day) using {d['model_used']}.")
        elif t == "get_improvement_actions":
            rows = [f"[{a['priority']}] {a['title']}: {a['action']}" + (f" (estimated exposure {a['estimated_exposure']})" if a["estimated_exposure"] else "") for a in d["actions"]]
            parts.append("**Recommended actions** (recommendations, not guarantees)\n" + _bullets(rows))
        for n in notes if t != "explain_profit_change" or "change" in d else []:
            parts.append(f"_Note: {n}_")
    return "\n\n".join(parts)


# --------------------------------------------------------------------------- analyst
class BusinessAnalyst:
    def __init__(self, analysis: Analysis, llm: LLMClient | None = None) -> None:
        self.analysis = analysis
        self.tools = BusinessTools(analysis)
        self.llm = llm

    def _plan_with_llm(self, question: str) -> list[tuple[str, dict[str, Any]]]:
        if self.llm is None:
            return []
        try:
            raw = self.llm.chat(prompts.planner_messages(question), temperature=0.0, max_tokens=200)
            m = re.search(r"\{.*\}", raw, re.S)
            plan = json.loads(m.group(0)) if m else {}
        except (LLMUnavailable, ValueError):
            return []
        calls = []
        for t in (plan.get("tools") or [])[:MAX_TOOLS]:
            try:
                calls.append((t["name"], validate_args(t["name"], t.get("args"))))
            except (KeyError, TypeError, ToolArgumentError):
                continue                    # invalid model-proposed calls are dropped, never executed
        return calls

    def ask(self, question: str, history: list[dict[str, str]] | None = None) -> AnalystAnswer:
        q = sanitize(question)
        if not q:
            return AnalystAnswer("Please type a question about your business.", "limitation")
        if self.analysis.ds.is_empty:
            return AnalystAnswer("No sales data is loaded, so I cannot answer yet. Load the demo data or upload your CSV files in Data Management.", "limitation")

        calls = route(q, self.tools) or self._plan_with_llm(q)
        if not calls:
            return AnalystAnswer("I could not match that question to the data I can analyse.\n\n" + HELP_TEXT, "limitation")

        results, used = [], []
        for name, args in calls:
            res = self.tools.call(name, args)
            used.append({"tool": name, "args": args, "ok": res.ok, "error": res.error})
            if res.ok:
                results.append({"tool": name, "args": args, "result": res.data})
        if not results:
            errs = "; ".join(u["error"] for u in used if u["error"])
            return AnalystAnswer(f"I could not calculate that: {errs}", "limitation", tools=used)

        facts_text = json.dumps(results, ensure_ascii=False, default=str)
        warnings: list[str] = []
        if self.llm is not None:
            try:
                text = self.llm.chat(prompts.answer_messages(q, facts_text, history), temperature=0.1)
                bad = ungrounded_numbers(text, facts_text, q)
                if bad:
                    warnings.append("Some figures in the AI wording could not be matched to the calculated data: " + ", ".join(bad[:6]) +
                                    ". Check the 'Calculated data used' panel before relying on them.")
                return AnalystAnswer(text, "llm", f"{self.llm.name}:{self.llm.model}", used, results, warnings, bad)
            except LLMUnavailable as e:
                warnings.append(f"AI model unavailable ({e}). Showing the fallback answer built from the calculated data.")
                log.warning("LLM unavailable, using fallback: %s", e)
        else:
            warnings.append("Fallback mode: no local language model is connected, so this answer was built from templates using the calculated data.")
        return AnalystAnswer(render_fallback(results), "fallback", "template", used, results, warnings)
