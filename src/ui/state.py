"""Cached, read-only access to the active dataset and analyses for Streamlit pages.

Streamlit reruns a page script on every interaction, so loading, analysis and model training are cached and keyed by
(dataset, data version, rules). Re-uploading data changes the version and invalidates the caches; nothing is retrained or
reinserted just because the page reran.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import pandas as pd
import streamlit as st

import config
from src.analytics.pipeline import Analysis, run_analysis
from src.data import loader
from src.data.repository import Repository
from src.genai.ollama_client import LLMClient, LLMStatus, HuggingFaceClient, OllamaClient, get_llm_client
from src.ml import forecasting as F
from src.utils.logging_config import get_logger

log = get_logger(__name__)
RULES_KEY = "business_rules"


@dataclass
class Ctx:
    repo: Repository
    ds: loader.Dataset
    analysis: Analysis
    rules: config.BusinessRules
    active: str
    version: str
    ml_demand: bool


@st.cache_resource
def get_repo() -> Repository:
    return Repository()


def load_rules(repo: Repository) -> config.BusinessRules:
    saved = repo.get_config(RULES_KEY, {}) or {}
    valid = {k: type(getattr(config.DEFAULT_RULES, k))(v) for k, v in saved.items() if hasattr(config.DEFAULT_RULES, k)}
    return config.BusinessRules(**{**asdict(config.DEFAULT_RULES), **valid})


@st.cache_resource(show_spinner="Loading your data...")
def _dataset(_repo: Repository, active: str, version: str) -> loader.Dataset:
    ds = _repo.load_dataset(active)
    return ds if ds is not None else loader.load_demo_dataset()


@st.cache_resource(show_spinner="Analysing your business data...")
def _analysis(_repo: Repository, _ds: loader.Dataset, active: str, version: str, rules_json: str, ml_demand: bool) -> Analysis:
    rules = config.BusinessRules(**json.loads(rules_json))
    override = _demand_override(_repo, _ds, active, version) if ml_demand else None
    an = run_analysis(_ds, rules, demand_override=override)
    try:
        _repo.log_analysis_run(active, an.headline() | {"ml_demand": bool(ml_demand)})
    except Exception:  # history is best-effort and must never break a page
        log.warning("Could not record analysis history")
    return an


@st.cache_resource(show_spinner=False)
def _demand_override_store() -> dict:
    return {}


def _demand_override(repo: Repository, ds: loader.Dataset, active: str, version: str) -> dict[str, float] | None:
    return _demand_override_store().get((active, version))


def compute_ml_demand(ctx: "Ctx", progress=None) -> int:
    """Train/validate forecasts for every product (horizon 7) and keep ML demand where the ML model beat the baselines."""
    pids = list(ctx.ds.inventory["product_id"])
    results = F.forecast_many(ctx.ds.sales, pids, 7, progress=progress)
    for pid, r in results.items():
        if r.sufficient and not r.metrics.empty:
            m = {"mae": r.metric(r.selected_model, "mae"), "wape": r.metric(r.selected_model, "wape"),
                 "baseline_mae": r.metric("recent_average", "mae"), "baseline_wape": r.metric("recent_average", "wape")}
            ctx.repo.save_forecast_meta(ctx.active, pid, 7, r.selected_model, m, ctx.version)
    override = F.demand_override_from(results)
    _demand_override_store()[(ctx.active, ctx.version)] = override
    return len(override)


def ml_demand_ready(active: str, version: str) -> bool:
    return (active, version) in _demand_override_store()


def ensure_ready() -> Ctx:
    """Called at the top of every page: makes sure data exists and returns the cached analysis context."""
    repo = get_repo()
    repo.ensure_demo_loaded()
    active = repo.get_active()
    version = repo.data_version(active)
    rules = load_rules(repo)
    ds = _dataset(repo, active, version)
    use_ml = bool(st.session_state.get("use_ml_demand")) and ml_demand_ready(active, version)
    an = _analysis(repo, ds, active, version, json.dumps(asdict(rules), sort_keys=True), use_ml)
    return Ctx(repo, ds, an, rules, active, version, use_ml)


@st.cache_data(show_spinner="Training and validating forecast models...", max_entries=300)
def cached_forecast(_ctx_ds: loader.Dataset, _repo: Repository, active: str, version: str, product_id: str, horizon: int) -> F.ForecastResult:
    res = F.forecast_product(_ctx_ds.sales, product_id, horizon)
    if res.sufficient and not res.metrics.empty:
        m = {"mae": res.metric(res.selected_model, "mae"), "wape": res.metric(res.selected_model, "wape"),
             "baseline_mae": res.metric("recent_average", "mae"), "baseline_wape": res.metric("recent_average", "wape")}
        _repo.save_forecast_meta(active, product_id, horizon, res.selected_model, m, version)
    return res


# --------------------------------------------------------------------------- LLM status / client
@st.cache_data(ttl=20, show_spinner=False)
def llm_statuses() -> list[LLMStatus]:
    return [OllamaClient().status(), HuggingFaceClient().status()]


def current_backend() -> str:
    return st.session_state.get("llm_backend", config.LLM_BACKEND)


def get_client() -> LLMClient | None:
    return get_llm_client(current_backend())


def llm_mode_summary() -> tuple[str, str]:
    """(short label, detail) describing which AI mode is active right now."""
    backend = current_backend()
    if backend == "fallback":
        return "Fallback mode", "AI model disabled in settings; answers use templates and calculated data."
    sts = {s.backend: s for s in llm_statuses()}
    order = ["ollama", "huggingface"] if backend == "auto" else [backend]
    for b in order:
        if b in sts and sts[b].available:
            return f"AI: {b} · {sts[b].model}", sts[b].message
    first = sts.get(order[0])
    return "Fallback mode", (first.message if first else "No local model available.")
