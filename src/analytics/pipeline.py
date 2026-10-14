"""Runs every deterministic analysis once and bundles the results (single source of truth for UI and AI tools)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

import config
from src.analytics import inventory as I
from src.analytics import profitability as P
from src.analytics import recommendations as R
from src.data.loader import Dataset
from src.ml import anomaly_detection as A


@dataclass
class Analysis:
    ds: Dataset
    rules: config.BusinessRules
    summary: P.FinancialSummary
    inventory: pd.DataFrame
    anomalies: pd.DataFrame
    recommendations: list[R.Recommendation]
    ran_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def inventory_counts(self) -> dict:
        return I.summary_counts(self.inventory)

    @property
    def anomaly_counts(self) -> dict:
        return A.summarize(self.anomalies)

    def headline(self) -> dict:
        ic, ac = self.inventory_counts, self.anomaly_counts
        return {"records": self.ds.record_count(), "revenue": self.summary.revenue, "gross_profit": self.summary.gross_profit,
                "low_stock_products": ic["low_stock"], "slow_moving_products": ic["slow_moving"], "anomalies": ac["total"],
                "recommendations": len(self.recommendations)}


def run_analysis(ds: Dataset, rules: config.BusinessRules | None = None, use_isolation_forest: bool = True,
                 demand_override: dict[str, float] | None = None) -> Analysis:
    rules = rules or config.DEFAULT_RULES
    inv = I.analyze_inventory(ds, rules, demand_override=demand_override)
    an = A.detect_anomalies(ds, rules, use_isolation_forest=use_isolation_forest)
    recs = R.build_recommendations(ds, inv, an, rules)
    return Analysis(ds, rules, P.compute_summary(ds.sales), inv, an, recs)
