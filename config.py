"""Central configuration for VyaparAI. All values can be overridden via environment / .env."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
ASSETS_DIR = BASE_DIR / "assets"

APP_NAME = "VyaparAI"
TAGLINE = "Turn Your Business Data Into Smarter Decisions."
CURRENCY_SYMBOL = "₹"
CURRENCY_CODE = "INR"

DB_PATH = Path(os.getenv("VYAPARAI_DB_PATH", str(DATA_DIR / "vyaparai.db")))
LOG_LEVEL = os.getenv("VYAPARAI_LOG_LEVEL", "INFO")

# ---- GenAI (free, local only) -------------------------------------------------
# Backend options: "auto" (Ollama, then Hugging Face local, then rule-based fallback),
# "ollama", "huggingface", "fallback".
LLM_BACKEND = os.getenv("VYAPARAI_LLM_BACKEND", "auto").lower()
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
# Free open-weight model downloaded from the Hugging Face Hub and run locally on CPU/GPU.
HF_MODEL = os.getenv("HF_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LLM_TIMEOUT_S = int(os.getenv("VYAPARAI_LLM_TIMEOUT", "120"))


@dataclass(frozen=True)
class BusinessRules:
    """Configurable thresholds. Defaults are documented in the README."""

    low_stock_service_level_z: float = 1.65      # ~95% service level
    target_coverage_days: int = 30               # reorder up to this many days of demand
    demand_window_days: int = 60                 # window for average daily demand
    slow_moving_daily_demand: float = 0.35       # units/day (about 1 unit every 3 days) below which stock is "slow"
    slow_moving_days_since_sale: int = 45
    dead_stock_days_since_sale: int = 120
    excess_stock_days_cover: int = 120
    low_margin_pct: float = 8.0
    high_discount_pct: float = 25.0
    large_qty_zscore: float = 4.0
    price_change_pct: float = 25.0
    spike_zscore: float = 3.5
    decline_pct: float = 30.0
    min_history_days: int = 28


DEFAULT_RULES = BusinessRules()
