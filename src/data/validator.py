"""CSV validation, column mapping and cleaning.

Principles
----------
* Problem rows are never silently discarded. Rows that cannot be analysed are moved to
  ``report.rejected`` with a reason; rows that are merely suspicious are kept and flagged.
* Unknown costs are never converted to zero. They stay NaN and are flagged (``cost_missing``).
* Monetary values are Indian Rupees (₹). Strings such as "₹1,200.50" or "Rs. 450" are parsed.

Conventions (also documented in the README)
-------------------------------------------
* ``discount`` is a PERCENTAGE (0-100) of quantity x unit_price. If a file stores discounts
  as rupee amounts per line, call ``validate_sales(..., discount_mode="amount")``.
* ``total_amount`` is the recorded net amount for the line (after discount). If it is blank it
  is computed as quantity x unit_price x (1 - discount/100) and flagged ``total_computed``.
* Negative ``quantity`` = customer return / refund. It is kept and reduces revenue and COGS.
* Exact duplicate lines are kept but flagged ``is_duplicate`` and excluded from metrics by default.
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd

from src.utils.dates import parse_dates

TOTAL_TOLERANCE_ABS = 1.0     # ₹
TOTAL_TOLERANCE_REL = 0.01    # 1 %


# ----------------------------------------------------------------------------- schema
@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: str                      # "str" | "date" | "num"
    required: bool
    description: str
    synonyms: tuple[str, ...] = ()
    fallback_note: str = ""        # what happens when the column is absent


SALES_SPEC: list[ColumnSpec] = [
    ColumnSpec("transaction_id", "str", False, "Unique id of the sales line/bill", ("txn_id", "bill_no", "invoice_no", "invoice", "order_id", "id"),
               "Ids are auto-generated; duplicate detection relies on row content only."),
    ColumnSpec("transaction_date", "date", True, "Date of sale", ("date", "sale_date", "invoice_date", "bill_date", "order_date")),
    ColumnSpec("product_id", "str", True, "Product / SKU code", ("sku", "item_id", "item_code", "product_code", "sku_id")),
    ColumnSpec("product_name", "str", False, "Product name", ("item", "item_name", "product", "description"),
               "product_id is used as the display name."),
    ColumnSpec("quantity", "num", True, "Units sold (negative = return)", ("qty", "units", "units_sold", "quantity_sold")),
    ColumnSpec("unit_price", "num", True, "Selling price per unit (₹)", ("price", "selling_price", "rate", "mrp", "sale_price")),
    ColumnSpec("unit_cost", "num", False, "Cost per unit (₹)", ("cost", "cost_price", "purchase_price", "cp"),
               "Cost is estimated from the inventory file where possible; otherwise margins are reported as unknown."),
    ColumnSpec("discount", "num", False, "Discount (% of line value)", ("discount_pct", "discount_percent", "disc", "discount_%"),
               "Discount is assumed to be 0%."),
    ColumnSpec("total_amount", "num", False, "Recorded net line amount (₹)", ("total", "amount", "line_total", "net_amount", "revenue", "sales_amount"),
               "Computed as quantity x unit_price x (1 - discount%)."),
    ColumnSpec("payment_method", "str", False, "Payment mode (UPI, Cash, Card...)", ("payment", "payment_mode", "mode_of_payment"),
               "Payment analysis is unavailable."),
    ColumnSpec("category", "str", False, "Product category", ("product_category", "department", "group", "cat"),
               "Category-level analysis groups everything under 'Uncategorised'."),
]

INVENTORY_SPEC: list[ColumnSpec] = [
    ColumnSpec("product_id", "str", True, "Product / SKU code", SALES_SPEC[2].synonyms),
    ColumnSpec("product_name", "str", False, "Product name", SALES_SPEC[3].synonyms, "product_id is used as the name."),
    ColumnSpec("category", "str", False, "Product category", SALES_SPEC[10].synonyms, "Category falls back to the sales file or 'Uncategorised'."),
    ColumnSpec("current_stock", "num", True, "Units currently in stock", ("stock", "qty_on_hand", "on_hand", "closing_stock", "quantity_in_stock", "stock_qty")),
    ColumnSpec("reorder_level", "num", False, "Reorder point (units)", ("reorder_point", "min_stock", "reorder_qty_level", "rol"),
               "A statistical reorder point is calculated from sales history."),
    ColumnSpec("supplier_lead_time_days", "num", False, "Supplier lead time (days)", ("lead_time", "lead_time_days", "supplier_lead_time"),
               "A default lead time (7 days) is assumed and flagged."),
    ColumnSpec("unit_cost", "num", False, "Latest cost per unit (₹)", ("cost", "cost_price", "purchase_price", "cp"),
               "Cost is estimated from sales history if available."),
    ColumnSpec("selling_price", "num", False, "Current selling price (₹)", ("price", "mrp", "sale_price", "unit_price")),
    ColumnSpec("last_sale_date", "date", False, "Date of last sale", ("last_sold", "last_sold_date"),
               "Derived from the sales file."),
]

PURCHASES_SPEC: list[ColumnSpec] = [
    ColumnSpec("purchase_date", "date", True, "Date goods were/will be received", ("date", "received_date", "restock_date", "order_date")),
    ColumnSpec("product_id", "str", True, "Product / SKU code", SALES_SPEC[2].synonyms),
    ColumnSpec("quantity_purchased", "num", True, "Units purchased", ("quantity", "qty", "units", "qty_purchased")),
    ColumnSpec("purchase_cost", "num", False, "Total purchase cost of the line (₹)", ("cost", "amount", "total_cost", "total"),
               "Purchase cost analysis is unavailable."),
]

SPECS = {"sales": SALES_SPEC, "inventory": INVENTORY_SPEC, "purchases": PURCHASES_SPEC}


# ----------------------------------------------------------------------------- report objects
@dataclass
class Issue:
    severity: str            # "error" | "warning" | "info"
    code: str
    message: str
    count: int = 0
    impact: str = ""
    sample: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {"severity": self.severity, "code": self.code, "message": self.message,
                "count": int(self.count), "impact": self.impact}


@dataclass
class ValidationReport:
    kind: str
    rows_in: int = 0
    rows_valid: int = 0
    issues: list[Issue] = field(default_factory=list)
    rejected: pd.DataFrame = field(default_factory=pd.DataFrame)
    mapping: dict[str, str | None] = field(default_factory=dict)

    @property
    def rows_rejected(self) -> int:
        return len(self.rejected)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)

    @property
    def status(self) -> str:
        if self.has_errors:
            return "Errors"
        if any(i.severity == "warning" for i in self.issues):
            return "Warnings"
        return "Clean"

    def add(self, severity: str, code: str, message: str, count: int = 0, impact: str = "",
            sample: pd.DataFrame | None = None) -> None:
        rows = sample.head(5).astype(str).to_dict("records") if sample is not None and len(sample) else []
        self.issues.append(Issue(severity, code, message, count, impact, rows))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([i.as_dict() for i in self.issues],
                            columns=["severity", "code", "message", "count", "impact"])

    def summary(self) -> dict:
        return {"kind": self.kind, "rows_in": self.rows_in, "rows_valid": self.rows_valid,
                "rows_rejected": self.rows_rejected, "status": self.status,
                "issues": [i.as_dict() for i in self.issues]}


# ----------------------------------------------------------------------------- column mapping
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9%]+", "_", str(s).strip().lower()).strip("_")


def suggest_mapping(columns: Iterable[str], spec: list[ColumnSpec]) -> dict[str, str | None]:
    """Map canonical column -> source column (or None). Exact, synonym, then fuzzy match."""
    cols = list(columns)
    normed = {c: _norm(c) for c in cols}
    used: set[str] = set()
    mapping: dict[str, str | None] = {}
    for sp in spec:  # pass 1: exact name, pass 2: synonyms
        mapping[sp.name] = next((c for c, n in normed.items() if n == sp.name and c not in used), None)
        if mapping[sp.name]:
            used.add(mapping[sp.name])  # type: ignore[arg-type]
    for sp in spec:
        if mapping[sp.name]:
            continue
        syn = {_norm(x) for x in sp.synonyms}
        hit = next((c for c, n in normed.items() if n in syn and c not in used), None)
        if hit:
            mapping[sp.name] = hit
            used.add(hit)
    for sp in spec:
        if mapping[sp.name]:
            continue
        pool = {c: n for c, n in normed.items() if c not in used}
        match = difflib.get_close_matches(sp.name, list(pool.values()), n=1, cutoff=0.8)
        if match:
            hit = next(c for c, n in pool.items() if n == match[0])
            mapping[sp.name] = hit
            used.add(hit)
    return mapping


def apply_mapping(df: pd.DataFrame, mapping: dict[str, str | None]) -> pd.DataFrame:
    """Return a frame with canonical column names (only mapped columns are kept)."""
    out = pd.DataFrame(index=df.index)
    for canon, src in mapping.items():
        if src and src in df.columns:
            out[canon] = df[src]
    return out


# ----------------------------------------------------------------------------- parsing helpers
_NON_NUM = re.compile(r"[^0-9.\-+eE]")


def to_numeric_clean(series: pd.Series) -> pd.Series:
    """Parse numbers, tolerating ₹, Rs., commas, spaces and (x) negatives. Unparseable -> NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.strip()
    neg = s.str.match(r"^\(.*\)$", na=False)
    s = s.str.replace(r"(?i)rs\.?|inr|₹", "", regex=True).str.replace(",", "", regex=False)
    s = s.str.replace(r"^\((.*)\)$", r"\1", regex=True).str.strip()
    out = pd.to_numeric(s, errors="coerce")
    out = out.where(~neg, -out)
    return out.astype("float64")


def _blank(series: pd.Series) -> pd.Series:
    return series.isna() | (series.astype("string").str.strip().fillna("") == "")


def _clean_str(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def _missing_columns(df: pd.DataFrame, spec: list[ColumnSpec], report: ValidationReport) -> bool:
    """Record missing-column issues. Returns True when a required column is missing (fatal)."""
    fatal = False
    for sp in spec:
        if sp.name in df.columns:
            continue
        if sp.required:
            fatal = True
            report.add("error", f"missing_required_{sp.name}",
                       f"Required column '{sp.name}' ({sp.description}) is missing.",
                       impact="This file cannot be analysed until the column is mapped or added.")
        else:
            report.add("warning", f"missing_{sp.name}", f"Column '{sp.name}' ({sp.description}) is missing.",
                       impact=sp.fallback_note or "Related analysis is limited.")
    return fatal


def _reject(raw: pd.DataFrame, mask: pd.Series, reason: str, bucket: list[pd.DataFrame]) -> None:
    if mask.any():
        r = raw.loc[mask].copy()
        r["reject_reason"] = reason
        r["source_row"] = r.index + 2  # +2: header line + 1-based
        bucket.append(r)


def _empty_rejected() -> pd.DataFrame:
    return pd.DataFrame(columns=["reject_reason", "source_row"])


# ----------------------------------------------------------------------------- sales
SALES_CLEAN_COLUMNS = [
    "transaction_id", "transaction_date", "product_id", "product_name", "quantity", "unit_price",
    "unit_cost", "discount", "total_amount", "expected_total", "payment_method", "category",
    "is_duplicate", "is_return", "cost_missing", "total_mismatch", "total_computed", "source_row",
]


def validate_sales(raw: pd.DataFrame, discount_mode: str = "percent") -> tuple[pd.DataFrame, ValidationReport]:
    """Validate and clean a sales frame that already uses canonical column names."""
    rep = ValidationReport("sales", rows_in=len(raw))
    if raw is None or len(raw) == 0:
        rep.add("error", "empty_file", "The sales file contains no rows.", impact="Nothing to analyse.")
        return _empty_sales(), rep
    df = raw.copy().reset_index(drop=True)
    if _missing_columns(df, SALES_SPEC, rep):
        rep.rejected = df.assign(reject_reason="missing required column(s)", source_row=df.index + 2)
        return _empty_sales(), rep

    rejects: list[pd.DataFrame] = []
    for col in ("transaction_id", "product_name", "payment_method", "category"):
        if col in df.columns:
            df[col] = _clean_str(df[col])
    df["product_id"] = _clean_str(df["product_id"])

    # --- dates
    parsed = parse_dates(df["transaction_date"])
    bad_date = parsed.isna()
    if bad_date.any():
        rep.add("error", "invalid_date", "Rows with missing or unparseable dates were set aside.", int(bad_date.sum()),
                "These rows are excluded from analysis (see rejected rows).", df.loc[bad_date])
    _reject(df, bad_date, "invalid transaction_date", rejects)

    # --- numerics
    qty = to_numeric_clean(df["quantity"])
    price = to_numeric_clean(df["unit_price"])
    bad_qty = qty.isna()
    bad_price = price.isna() | (price < 0)
    zero_qty = (qty == 0) & ~bad_qty
    missing_pid = df["product_id"].isna() | (df["product_id"] == "")
    for mask, code, msg in [
        (bad_qty & ~bad_date, "invalid_quantity", "Rows with non-numeric or blank quantity were set aside."),
        (bad_price & ~bad_date, "invalid_price", "Rows with non-numeric, blank or negative unit_price were set aside."),
        (zero_qty & ~bad_date, "zero_quantity", "Rows with quantity = 0 were set aside."),
        (missing_pid & ~bad_date, "missing_product_id", "Rows without a product_id were set aside."),
    ]:
        if mask.any():
            rep.add("error", code, msg, int(mask.sum()), "Excluded from analysis (see rejected rows).", df.loc[mask])
    _reject(df, bad_qty & ~bad_date, "invalid quantity", rejects)
    _reject(df, bad_price & ~bad_date, "invalid unit_price", rejects)
    _reject(df, zero_qty & ~bad_date, "zero quantity", rejects)
    _reject(df, missing_pid & ~bad_date, "missing product_id", rejects)

    # --- discount
    if "discount" in df.columns:
        disc_raw = df["discount"].astype("string").str.replace("%", "", regex=False)
        disc = to_numeric_clean(disc_raw)
        blank = _blank(df["discount"])
        junk = disc.isna() & ~blank
        if junk.any():
            rep.add("warning", "invalid_discount", "Some discount values are not numeric and were treated as 0%.", int(junk.sum()),
                    "Revenue per line is taken from total_amount where recorded.", df.loc[junk])
        disc = disc.fillna(0.0)
        if discount_mode == "amount":
            line_value = (qty.abs() * price).replace(0, np.nan)
            disc = (disc.abs() / line_value * 100).fillna(0.0)
            rep.add("info", "discount_amount_mode", "Discount values were interpreted as rupee amounts per line and converted to %.", 0)
    else:
        disc = pd.Series(0.0, index=df.index)
    out_of_range = (disc < 0) | (disc > 100)
    if out_of_range.any():
        rep.add("error", "discount_out_of_range", "Discounts outside 0-100% were set aside.", int(out_of_range.sum()),
                "Excluded from analysis (see rejected rows).", df.loc[out_of_range])
    _reject(df, out_of_range & ~bad_date, "discount outside 0-100%", rejects)

    # --- cost
    if "unit_cost" in df.columns:
        cost = to_numeric_clean(df["unit_cost"])
        neg_cost = cost < 0
        if neg_cost.any():
            rep.add("warning", "negative_cost", "Negative unit_cost values were treated as unknown.", int(neg_cost.sum()),
                    "Margin for these rows is unknown.", df.loc[neg_cost])
            cost = cost.where(~neg_cost)
    else:
        cost = pd.Series(np.nan, index=df.index)

    expected = (qty * price * (1 - disc / 100)).round(2)
    if "total_amount" in df.columns:
        total = to_numeric_clean(df["total_amount"])
        total_computed = total.isna()
    else:
        total = pd.Series(np.nan, index=df.index)
        total_computed = pd.Series(True, index=df.index)
    total = total.where(~total_computed, expected)
    tol = np.maximum(TOTAL_TOLERANCE_ABS, TOTAL_TOLERANCE_REL * expected.abs())
    mismatch = (~total_computed) & ((total - expected).abs() > tol)

    reject_idx = pd.Index([])
    if rejects:
        reject_idx = pd.Index(sorted(set().union(*[set(r.index) for r in rejects])))
    keep = ~df.index.isin(reject_idx)

    clean = pd.DataFrame({
        "transaction_id": df["transaction_id"] if "transaction_id" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string"),
        "transaction_date": parsed.dt.normalize(),
        "product_id": df["product_id"],
        "product_name": df["product_name"] if "product_name" in df.columns else df["product_id"],
        "quantity": qty, "unit_price": price, "unit_cost": cost, "discount": disc,
        "total_amount": total, "expected_total": expected,
        "payment_method": df["payment_method"] if "payment_method" in df.columns else pd.Series("Unknown", index=df.index),
        "category": df["category"] if "category" in df.columns else pd.Series("Uncategorised", index=df.index),
        "is_return": qty < 0, "cost_missing": cost.isna(),
        "total_mismatch": mismatch, "total_computed": total_computed,
        "source_row": df.index + 2,
    })[keep].copy()

    clean["product_name"] = clean["product_name"].where(~_blank(clean["product_name"]), clean["product_id"])
    clean["category"] = clean["category"].where(~_blank(clean["category"]), "Uncategorised")
    clean["payment_method"] = clean["payment_method"].where(~_blank(clean["payment_method"]), "Unknown")

    # --- ids and duplicates
    if "transaction_id" not in df.columns or _blank(clean["transaction_id"]).all():
        clean["transaction_id"] = [f"AUTO-{i + 1:07d}" for i in range(len(clean))]
    else:
        blank_id = _blank(clean["transaction_id"])
        if blank_id.any():
            rep.add("warning", "missing_transaction_id", "Some rows have no transaction_id; ids were generated.", int(blank_id.sum()),
                    "Id-based duplicate detection is not possible for those rows.")
            clean.loc[blank_id, "transaction_id"] = [f"AUTO-{i + 1:07d}" for i in range(int(blank_id.sum()))]
    key = ["transaction_id", "transaction_date", "product_id", "quantity", "unit_price", "discount", "total_amount"]
    clean["is_duplicate"] = clean.duplicated(subset=key, keep="first")
    n_dup = int(clean["is_duplicate"].sum())
    if n_dup:
        rep.add("warning", "duplicate_transactions", "Exact duplicate transaction lines were found.", n_dup,
                "Duplicates are kept but flagged and excluded from revenue/profit metrics by default.",
                clean[clean["is_duplicate"]])
    conflict = clean.duplicated(subset=["transaction_id", "product_id"], keep=False) & ~clean["is_return"]
    conflict &= ~clean.duplicated(subset=key, keep=False)
    if conflict.any():
        rep.add("warning", "conflicting_transaction_id", "The same transaction_id and product appear with different details.",
                int(conflict.sum()), "Not excluded; verify these lines are not entry errors.", clean[conflict])

    # --- warnings on kept rows
    if clean["is_return"].any():
        rep.add("info", "returns_present", "Negative quantities were treated as customer returns.", int(clean["is_return"].sum()),
                "They reduce recognised revenue and cost of goods sold.", clean[clean["is_return"]])
    if clean["cost_missing"].any():
        rep.add("warning", "missing_unit_cost", "Some rows have no unit_cost.", int(clean["cost_missing"].sum()),
                "Cost is estimated from the inventory file where possible; remaining rows are excluded from margin figures.",
                clean[clean["cost_missing"]])
    if clean["total_mismatch"].any():
        diff = (clean.loc[clean["total_mismatch"], "total_amount"] - clean.loc[clean["total_mismatch"], "expected_total"]).abs().sum()
        rep.add("warning", "inconsistent_total", "total_amount differs from quantity x price x (1 - discount).",
                int(clean["total_mismatch"].sum()),
                f"Recorded totals are used; the absolute difference is ₹{diff:,.2f}. Verify these lines.",
                clean[clean["total_mismatch"]])
    zero_price = (clean["unit_price"] == 0)
    if zero_price.any():
        rep.add("warning", "zero_price", "Rows with unit_price = 0 (free items?).", int(zero_price.sum()), "Kept; revenue is zero for these lines.", clean[zero_price])
    below_cost = (clean["unit_price"] < clean["unit_cost"]) & ~clean["is_return"]
    if below_cost.any():
        rep.add("info", "sold_below_cost", "Rows sold below their recorded unit cost.", int(below_cost.sum()),
                "Reported in Profit Leakage Detection.", clean[below_cost])

    rep.rejected = pd.concat(rejects).loc[lambda x: ~x.index.duplicated(keep="first")].sort_index() if rejects else _empty_rejected()
    rep.rows_valid = len(clean)
    if len(clean) == 0:
        rep.add("error", "no_valid_rows", "No valid rows remain after validation.", impact="Nothing to analyse.")
    return clean.reset_index(drop=True)[SALES_CLEAN_COLUMNS], rep


def _empty_sales() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in SALES_CLEAN_COLUMNS})
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    for c in ("quantity", "unit_price", "unit_cost", "discount", "total_amount", "expected_total"):
        df[c] = df[c].astype("float64")
    for c in ("is_duplicate", "is_return", "cost_missing", "total_mismatch", "total_computed"):
        df[c] = df[c].astype(bool)
    return df


# ----------------------------------------------------------------------------- inventory
INVENTORY_CLEAN_COLUMNS = [
    "product_id", "product_name", "category", "current_stock", "reorder_level", "supplier_lead_time_days",
    "unit_cost", "selling_price", "last_sale_date", "negative_stock", "source_row",
]


def validate_inventory(raw: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport("inventory", rows_in=0 if raw is None else len(raw))
    if raw is None or len(raw) == 0:
        rep.add("error", "empty_file", "The inventory file contains no rows.", impact="Inventory analysis is unavailable.")
        return _empty_inventory(), rep
    df = raw.copy().reset_index(drop=True)
    if _missing_columns(df, INVENTORY_SPEC, rep):
        rep.rejected = df.assign(reject_reason="missing required column(s)", source_row=df.index + 2)
        return _empty_inventory(), rep

    rejects: list[pd.DataFrame] = []
    df["product_id"] = _clean_str(df["product_id"])
    missing_pid = df["product_id"].isna() | (df["product_id"] == "")
    stock = to_numeric_clean(df["current_stock"])
    bad_stock = stock.isna()
    if missing_pid.any():
        rep.add("error", "missing_product_id", "Rows without a product_id were set aside.", int(missing_pid.sum()), "Excluded from analysis.", df.loc[missing_pid])
    if bad_stock.any():
        rep.add("error", "invalid_stock", "Rows with non-numeric or blank current_stock were set aside.", int(bad_stock.sum()),
                "Excluded; those products will not appear in inventory analysis.", df.loc[bad_stock])
    _reject(df, missing_pid, "missing product_id", rejects)
    _reject(df, bad_stock & ~missing_pid, "invalid current_stock", rejects)
    keep = ~(missing_pid | bad_stock)

    def num_col(name: str, neg_ok: bool = False) -> pd.Series:
        if name not in df.columns:
            return pd.Series(np.nan, index=df.index)
        s = to_numeric_clean(df[name])
        if not neg_ok:
            bad = s < 0
            if bad.any():
                rep.add("warning", f"negative_{name}", f"Negative {name} values were treated as unknown.", int(bad.sum()), "", df.loc[bad])
                s = s.where(~bad)
        return s

    lead = num_col("supplier_lead_time_days")
    clean = pd.DataFrame({
        "product_id": df["product_id"],
        "product_name": _clean_str(df["product_name"]) if "product_name" in df.columns else df["product_id"],
        "category": _clean_str(df["category"]) if "category" in df.columns else pd.Series("Uncategorised", index=df.index),
        "current_stock": stock, "reorder_level": num_col("reorder_level"),
        "supplier_lead_time_days": lead, "unit_cost": num_col("unit_cost"), "selling_price": num_col("selling_price"),
        "last_sale_date": parse_dates(df["last_sale_date"]).dt.normalize() if "last_sale_date" in df.columns else pd.NaT,
        "negative_stock": stock < 0, "source_row": df.index + 2,
    })[keep].copy()
    clean["product_name"] = clean["product_name"].where(~_blank(clean["product_name"]), clean["product_id"])
    clean["category"] = clean["category"].where(~_blank(clean["category"]), "Uncategorised")

    if clean["negative_stock"].any():
        rep.add("warning", "negative_stock", "Products with negative current_stock.", int(clean["negative_stock"].sum()),
                "Kept and flagged: stock records are probably out of date (sales recorded before receipts).", clean[clean["negative_stock"]])
    dup = clean.duplicated("product_id", keep="last")
    if dup.any():
        rep.add("warning", "duplicate_product_id", "Duplicate product_id rows; the last occurrence was kept.", int(dup.sum()),
                "Earlier duplicates are listed in rejected rows.", clean[dup])
        r = clean[dup].copy()
        r["reject_reason"] = "duplicate product_id (older row)"
        rejects.append(r)
        clean = clean[~dup]
    if "unit_cost" in df.columns and clean["unit_cost"].isna().any():
        rep.add("warning", "missing_unit_cost", "Some products have no unit_cost.", int(clean["unit_cost"].isna().sum()),
                "Inventory value for them is unknown; cost may be estimated from sales.", clean[clean["unit_cost"].isna()])
    under = (clean["selling_price"] < clean["unit_cost"])
    if under.any():
        rep.add("warning", "price_below_cost", "Products whose selling_price is below unit_cost.", int(under.sum()),
                "Each sale loses money at the listed price.", clean[under])

    rep.rejected = pd.concat(rejects, sort=False).sort_index() if rejects else _empty_rejected()
    rep.rows_valid = len(clean)
    return clean.reset_index(drop=True)[INVENTORY_CLEAN_COLUMNS], rep


def _empty_inventory() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in INVENTORY_CLEAN_COLUMNS})
    for c in ("current_stock", "reorder_level", "supplier_lead_time_days", "unit_cost", "selling_price"):
        df[c] = df[c].astype("float64")
    df["last_sale_date"] = pd.to_datetime(df["last_sale_date"])
    df["negative_stock"] = df["negative_stock"].astype(bool)
    return df


# ----------------------------------------------------------------------------- purchases
PURCHASE_CLEAN_COLUMNS = ["purchase_date", "product_id", "quantity_purchased", "purchase_cost", "source_row"]


def validate_purchases(raw: pd.DataFrame | None) -> tuple[pd.DataFrame, ValidationReport]:
    rep = ValidationReport("purchases", rows_in=0 if raw is None else len(raw))
    if raw is None or len(raw) == 0:
        rep.add("info", "no_purchases", "No purchase records were provided.", impact="Incoming stock and stock reconciliation are unavailable.")
        return _empty_purchases(), rep
    df = raw.copy().reset_index(drop=True)
    if _missing_columns(df, PURCHASES_SPEC, rep):
        rep.rejected = df.assign(reject_reason="missing required column(s)", source_row=df.index + 2)
        return _empty_purchases(), rep
    rejects: list[pd.DataFrame] = []
    d = parse_dates(df["purchase_date"])
    pid = _clean_str(df["product_id"])
    q = to_numeric_clean(df["quantity_purchased"])
    bad_date, bad_q = d.isna(), q.isna() | (q <= 0)
    bad_pid = pid.isna() | (pid == "")
    for mask, code, msg in [(bad_date, "invalid_date", "Purchase rows with invalid dates were set aside."),
                            (bad_q & ~bad_date, "invalid_quantity", "Purchase rows with non-numeric or non-positive quantity were set aside."),
                            (bad_pid & ~bad_date, "missing_product_id", "Purchase rows without product_id were set aside.")]:
        if mask.any():
            rep.add("error", code, msg, int(mask.sum()), "Excluded from analysis.", df.loc[mask])
    _reject(df, bad_date, "invalid purchase_date", rejects)
    _reject(df, bad_q & ~bad_date, "invalid quantity_purchased", rejects)
    _reject(df, bad_pid & ~bad_date & ~bad_q, "missing product_id", rejects)
    keep = ~(bad_date | bad_q | bad_pid)
    cost = to_numeric_clean(df["purchase_cost"]) if "purchase_cost" in df.columns else pd.Series(np.nan, index=df.index)
    clean = pd.DataFrame({"purchase_date": d.dt.normalize(), "product_id": pid, "quantity_purchased": q,
                          "purchase_cost": cost, "source_row": df.index + 2})[keep]
    rep.rejected = pd.concat(rejects).loc[lambda x: ~x.index.duplicated(keep="first")].sort_index() if rejects else _empty_rejected()
    rep.rows_valid = len(clean)
    return clean.reset_index(drop=True)[PURCHASE_CLEAN_COLUMNS], rep


def _empty_purchases() -> pd.DataFrame:
    df = pd.DataFrame({c: pd.Series(dtype="object") for c in PURCHASE_CLEAN_COLUMNS})
    df["purchase_date"] = pd.to_datetime(df["purchase_date"])
    for c in ("quantity_purchased", "purchase_cost"):
        df[c] = df[c].astype("float64")
    return df


# ----------------------------------------------------------------------------- cross-file checks
def cross_validate(sales: pd.DataFrame, inventory: pd.DataFrame, purchases: pd.DataFrame) -> list[Issue]:
    issues: list[Issue] = []
    if len(sales) and len(inventory):
        s_ids, i_ids = set(sales["product_id"]), set(inventory["product_id"])
        only_sales = s_ids - i_ids
        if only_sales:
            issues.append(Issue("warning", "sales_products_not_in_inventory",
                                f"{len(only_sales)} product(s) appear in sales but not in the inventory file.", len(only_sales),
                                "They are excluded from stock-level analysis.", [{"product_id": p} for p in sorted(only_sales)[:5]]))
        no_sales = i_ids - s_ids
        if no_sales:
            issues.append(Issue("info", "inventory_products_without_sales",
                                f"{len(no_sales)} inventory product(s) have no sales history.", len(no_sales),
                                "Demand cannot be estimated for them; they may be new or unsellable.", [{"product_id": p} for p in sorted(no_sales)[:5]]))
    if len(purchases) and len(inventory):
        unknown = set(purchases["product_id"]) - set(inventory["product_id"])
        if unknown:
            issues.append(Issue("warning", "purchases_unknown_products",
                                f"{len(unknown)} purchased product(s) are missing from the inventory file.", len(unknown), "", []))
    return issues
