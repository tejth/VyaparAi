import pandas as pd
import pytest

import config
from src.ml import anomaly_detection as A
from tests.conftest import dataset_from, daily_sales, make_inventory


def base_sales(days=60):
    return daily_sales(days, 2)


def detect(sales, **kw):
    return A.detect_anomalies(dataset_from(sales), use_isolation_forest=False, **kw)


def kinds(df):
    return set(df["detection"])


def test_clean_data_has_no_findings():
    assert detect(base_sales()).empty


def test_duplicate_flagged_as_data_quality():
    s = base_sales()
    s = pd.concat([s, s.iloc[[5]]], ignore_index=True)
    out = detect(s)
    d = out[out["detection"] == "Duplicate transaction"]
    assert len(d) == 1 and d.iloc[0]["group"] == "Data quality" and d.iloc[0]["severity"] == "Medium"


def test_inconsistent_total():
    s = base_sales()
    s.loc[3, "total_amount"] = "50"
    out = detect(s)
    r = out[out["detection"] == "Inconsistent total"].iloc[0]
    assert r["group"] == "Data quality" and r["amount_inr"] == pytest.approx(150)


def test_large_discount_flagged_and_marked_business_risk_not_fraud():
    s = base_sales()
    s.loc[10, ["discount", "total_amount"]] = ["80", "40"]
    out = detect(s)
    r = out[out["detection"] == "Unusually large discount"].iloc[0]
    assert r["group"] == "Business risk" and r["severity"] == "High" and "fraud" not in (r["explanation"] + r["next_step"]).lower()
    assert r["expected"].startswith("Below")


def test_large_quantity_and_price_spike_are_statistical_anomalies():
    s = base_sales(80)
    s.loc[20, ["quantity", "total_amount"]] = ["200", "20000"]
    s.loc[40, ["unit_price", "total_amount"]] = ["1000", "2000"]
    out = detect(s)
    assert "Abnormally large quantity" in kinds(out) and "Unexpected price change" in kinds(out)
    assert set(out[out["detection"].isin(["Abnormally large quantity", "Unexpected price change"])]["group"]) == {"Statistical anomaly"}


def test_negative_margin_and_missing_cost():
    s = base_sales()
    s["unit_cost"] = "120"                                 # selling at 100 with cost 120
    out = A.detect_anomalies(dataset_from(s), use_isolation_forest=False)
    r = out[out["detection"] == "Negative gross margin"].iloc[0]
    assert r["severity"] == "High" and r["amount_inr"] > 0
    s = base_sales()
    s.loc[:, "unit_cost"] = ""
    assert "Missing unit cost" in kinds(detect(s))


def test_sales_decline_detected():
    s = base_sales(70)
    s.loc[40:, ["quantity", "total_amount"]] = ["0.5", "50"]
    s["quantity"] = s["quantity"].astype(float).round().clip(lower=1).astype(int).astype(str)
    s.loc[40:, "quantity"] = "1"
    out = detect(s)
    assert "Sudden sales decline" in kinds(out)


def test_every_finding_has_required_fields(demo_analysis):
    an = demo_analysis.anomalies
    assert len(an) > 50
    for col in ["record_id", "date", "detection", "observed", "expected", "severity", "explanation", "next_step", "group"]:
        assert an[col].notna().all(), col
    assert set(an["severity"]) <= {"High", "Medium", "Low"}
    assert set(an["group"]) == {"Data quality", "Statistical anomaly", "Business risk"}


def test_injected_demo_anomalies_are_found(demo_analysis):
    c = demo_analysis.anomalies.groupby("detection").size()
    assert c["Duplicate transaction"] == 8 and c["Inconsistent total"] == 4
    assert c.get("Unexpected price change", 0) == 3 and c["Unusually large discount"] >= 6
    assert c["Abnormally large quantity"] >= 4 and "Negative gross margin" in c


def test_thresholds_are_configurable():
    s = base_sales()
    s.loc[10, ["discount", "total_amount"]] = ["30", "140"]
    strict = detect(s, rules=config.BusinessRules(high_discount_pct=20))
    lax = detect(s, rules=config.BusinessRules(high_discount_pct=50))
    assert "Unusually large discount" in kinds(strict) and "Unusually large discount" not in kinds(lax)


def test_isolation_forest_adds_only_low_severity_extras(demo_ds):
    with_if = A.detect_anomalies(demo_ds, use_isolation_forest=True)
    without = A.detect_anomalies(demo_ds, use_isolation_forest=False)
    extra = with_if[with_if["detection"] == "Isolation Forest outlier"]
    assert len(with_if) == len(without) + len(extra) and set(extra["severity"]) <= {"Low"}


def test_empty_dataset():
    assert A.detect_anomalies(dataset_from(pd.DataFrame())).empty
    assert A.summarize(pd.DataFrame())["total"] == 0
