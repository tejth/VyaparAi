"""SQLite persistence (stdlib sqlite3, parameterised statements only).

Demo data and uploaded data live in the same tables but are separated by a ``dataset`` column
("demo" / "uploaded"); exactly one is *active* at a time. Nothing here is ever built from
model-generated SQL.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

import config
from src.data import loader, validator as V
from src.utils.logging_config import get_logger

log = get_logger(__name__)

DATASETS = (loader.DEMO_LABEL, loader.UPLOADED_LABEL)

_SALES_COLS = ["transaction_id", "transaction_date", "product_id", "product_name", "quantity", "unit_price",
               "unit_cost", "discount", "total_amount", "expected_total", "payment_method", "category",
               "is_duplicate", "is_return", "cost_missing", "total_mismatch", "total_computed", "source_row"]
_INV_COLS = ["product_id", "product_name", "category", "current_stock", "reorder_level", "supplier_lead_time_days",
             "unit_cost", "selling_price", "last_sale_date", "negative_stock", "source_row"]
_PUR_COLS = ["purchase_date", "product_id", "quantity_purchased", "purchase_cost", "source_row"]
_BOOL_SALES = ["is_duplicate", "is_return", "cost_missing", "total_mismatch", "total_computed"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS sales (
    dataset TEXT NOT NULL, transaction_id TEXT, transaction_date TEXT, product_id TEXT, product_name TEXT,
    quantity REAL, unit_price REAL, unit_cost REAL, discount REAL, total_amount REAL, expected_total REAL,
    payment_method TEXT, category TEXT, is_duplicate INTEGER, is_return INTEGER, cost_missing INTEGER,
    total_mismatch INTEGER, total_computed INTEGER, source_row INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sales_ds ON sales(dataset, transaction_date);
CREATE TABLE IF NOT EXISTS inventory (
    dataset TEXT NOT NULL, product_id TEXT, product_name TEXT, category TEXT, current_stock REAL, reorder_level REAL,
    supplier_lead_time_days REAL, unit_cost REAL, selling_price REAL, last_sale_date TEXT, negative_stock INTEGER,
    source_row INTEGER
);
CREATE INDEX IF NOT EXISTS idx_inv_ds ON inventory(dataset, product_id);
CREATE TABLE IF NOT EXISTS purchases (
    dataset TEXT NOT NULL, purchase_date TEXT, product_id TEXT, quantity_purchased REAL, purchase_cost REAL,
    source_row INTEGER
);
CREATE INDEX IF NOT EXISTS idx_pur_ds ON purchases(dataset, product_id);
CREATE TABLE IF NOT EXISTS rejected_rows (
    dataset TEXT NOT NULL, kind TEXT NOT NULL, source_row INTEGER, reason TEXT, row_json TEXT
);
CREATE TABLE IF NOT EXISTS dataset_meta (
    dataset TEXT PRIMARY KEY, loaded_at TEXT NOT NULL, reports_json TEXT NOT NULL, label TEXT
);
CREATE TABLE IF NOT EXISTS app_config (key TEXT PRIMARY KEY, value_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ran_at TEXT NOT NULL, dataset TEXT NOT NULL, summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS forecast_meta (
    dataset TEXT NOT NULL, product_id TEXT NOT NULL, horizon INTEGER NOT NULL, model TEXT NOT NULL,
    mae REAL, wape REAL, baseline_mae REAL, baseline_wape REAL, trained_at TEXT NOT NULL, data_version TEXT,
    PRIMARY KEY (dataset, product_id, horizon)
);
"""


class Repository:
    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = Path(db_path or config.DB_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_schema()

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.db_path, timeout=30)
        try:
            yield con
            con.commit()
        except Exception:
            con.rollback()
            raise
        finally:
            con.close()

    def init_schema(self) -> None:
        with self.connect() as con:
            con.executescript(SCHEMA)

    @staticmethod
    def _check(dataset: str) -> str:
        if dataset not in DATASETS:
            raise ValueError(f"Unknown dataset '{dataset}'")
        return dataset

    # ------------------------------------------------------------------ datasets
    def save_dataset(self, ds: loader.Dataset, dataset: str | None = None) -> None:
        """Replace the stored rows for a dataset label inside one transaction."""
        label = self._check(dataset or ds.source)
        with self.connect() as con:
            for table in ("sales", "inventory", "purchases", "rejected_rows"):
                con.execute(f"DELETE FROM {table} WHERE dataset = ?", (label,))  # table names are constants
            self._insert(con, "sales", label, ds.sales, _SALES_COLS)
            self._insert(con, "inventory", label, ds.inventory, _INV_COLS)
            self._insert(con, "purchases", label, ds.purchases, _PUR_COLS)
            for kind, rep in ds.reports.items():
                rej = rep.rejected
                if len(rej):
                    rows = [(label, kind, int(r.get("source_row", 0) or 0), str(r.get("reject_reason", "")),
                             json.dumps({k: (None if pd.isna(v) else str(v)) for k, v in r.items()
                                         if k not in ("reject_reason", "source_row")}))
                            for r in rej.to_dict("records")]
                    con.executemany("INSERT INTO rejected_rows VALUES (?,?,?,?,?)", rows)
            payload = {k: r.summary() for k, r in ds.reports.items()}
            payload["_cross"] = [i.as_dict() for i in ds.cross_issues]
            con.execute("INSERT OR REPLACE INTO dataset_meta(dataset, loaded_at, reports_json, label) VALUES (?,?,?,?)",
                        (label, datetime.now().isoformat(timespec="seconds"), json.dumps(payload), label))
        log.info("Saved dataset '%s': %d sales, %d inventory, %d purchases rows", label, len(ds.sales),
                 len(ds.inventory), len(ds.purchases))

    @staticmethod
    def _insert(con: sqlite3.Connection, table: str, label: str, df: pd.DataFrame, cols: list[str]) -> None:
        if df is None or len(df) == 0:
            return
        out = pd.DataFrame({c: df[c] if c in df.columns else None for c in cols})
        for c in out.columns:
            if pd.api.types.is_datetime64_any_dtype(out[c]):
                out[c] = out[c].dt.strftime("%Y-%m-%d")
            elif pd.api.types.is_bool_dtype(out[c]):
                out[c] = out[c].astype(int)
        out = out.astype(object).where(out.notna(), None)
        rows = [(label, *r) for r in out.itertuples(index=False, name=None)]
        placeholders = ",".join("?" * (len(cols) + 1))
        con.executemany(f"INSERT INTO {table} (dataset, {', '.join(cols)}) VALUES ({placeholders})",  # noqa: S608
                        [tuple(v.item() if isinstance(v, np.generic) else (str(v) if isinstance(v, pd.Timestamp) else v) for v in row)
                         for row in rows])

    def has_dataset(self, dataset: str) -> bool:
        with self.connect() as con:
            return con.execute("SELECT 1 FROM dataset_meta WHERE dataset = ?", (self._check(dataset),)).fetchone() is not None

    def load_dataset(self, dataset: str) -> loader.Dataset | None:
        label = self._check(dataset)
        with self.connect() as con:
            meta = con.execute("SELECT reports_json FROM dataset_meta WHERE dataset = ?", (label,)).fetchone()
            if meta is None:
                return None
            sales = pd.read_sql_query("SELECT * FROM sales WHERE dataset = ?", con, params=(label,))
            inv = pd.read_sql_query("SELECT * FROM inventory WHERE dataset = ?", con, params=(label,))
            pur = pd.read_sql_query("SELECT * FROM purchases WHERE dataset = ?", con, params=(label,))
            rej = pd.read_sql_query("SELECT kind, source_row, reason, row_json FROM rejected_rows WHERE dataset = ?", con, params=(label,))
        sales = self._typed(sales.drop(columns=["dataset"]), "sales")
        inv = self._typed(inv.drop(columns=["dataset"]), "inventory")
        pur = self._typed(pur.drop(columns=["dataset"]), "purchases")
        payload = json.loads(meta[0])
        reports: dict[str, V.ValidationReport] = {}
        for kind in ("sales", "inventory", "purchases"):
            s = payload.get(kind, {})
            rep = V.ValidationReport(kind, rows_in=s.get("rows_in", 0), rows_valid=s.get("rows_valid", 0))
            rep.issues = [V.Issue(i["severity"], i["code"], i["message"], i.get("count", 0), i.get("impact", "")) for i in s.get("issues", [])]
            r = rej[rej["kind"] == kind]
            if len(r):
                rep.rejected = pd.DataFrame(
                    [{**json.loads(x.row_json), "reject_reason": x.reason, "source_row": x.source_row} for x in r.itertuples()])
            reports[kind] = rep
        ds = loader.finalize(sales, inv, pur, reports, label)
        ds.cross_issues = [V.Issue(i["severity"], i["code"], i["message"], i.get("count", 0), i.get("impact", ""))
                           for i in payload.get("_cross", [])]
        return ds

    @staticmethod
    def _typed(df: pd.DataFrame, kind: str) -> pd.DataFrame:
        for c in ("transaction_date", "purchase_date", "last_sale_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
        for c in _BOOL_SALES + ["negative_stock"]:
            if c in df.columns:
                df[c] = df[c].fillna(0).astype(bool)
        for c in df.columns:
            if df[c].dtype == object and c not in ("source_row",):
                df[c] = df[c].astype("string")
        if kind == "sales":
            df["source_row"] = df["source_row"].astype("int64")
        return df.reset_index(drop=True)

    def delete_dataset(self, dataset: str) -> None:
        label = self._check(dataset)
        with self.connect() as con:
            for table in ("sales", "inventory", "purchases", "rejected_rows", "forecast_meta"):
                con.execute(f"DELETE FROM {table} WHERE dataset = ?", (label,))
            con.execute("DELETE FROM dataset_meta WHERE dataset = ?", (label,))

    def dataset_info(self) -> dict[str, dict[str, Any]]:
        info: dict[str, dict[str, Any]] = {}
        with self.connect() as con:
            for label in DATASETS:
                row = con.execute("SELECT loaded_at FROM dataset_meta WHERE dataset = ?", (label,)).fetchone()
                if row:
                    n = con.execute("SELECT COUNT(*) FROM sales WHERE dataset = ?", (label,)).fetchone()[0]
                    info[label] = {"loaded_at": row[0], "sales_rows": n}
        return info

    def data_version(self, dataset: str) -> str:
        with self.connect() as con:
            row = con.execute("SELECT loaded_at FROM dataset_meta WHERE dataset = ?", (self._check(dataset),)).fetchone()
        return row[0] if row else "none"

    # ------------------------------------------------------------------ active dataset / demo
    def get_active(self) -> str:
        active = self.get_config("active_dataset", loader.DEMO_LABEL)
        return active if active in DATASETS and self.has_dataset(active) else loader.DEMO_LABEL

    def set_active(self, dataset: str) -> None:
        if not self.has_dataset(self._check(dataset)):
            raise ValueError(f"Dataset '{dataset}' has not been loaded")
        self.set_config("active_dataset", dataset)

    def ensure_demo_loaded(self) -> None:
        """Idempotent: loads demo data only when it is not already stored (no duplicate inserts on rerun)."""
        if not self.has_dataset(loader.DEMO_LABEL):
            self.reset_demo()

    def reset_demo(self) -> None:
        """Reload the synthetic demo data and make it active. Uploaded data is left untouched."""
        self.save_dataset(loader.load_demo_dataset(), loader.DEMO_LABEL)
        self.set_config("active_dataset", loader.DEMO_LABEL)

    # ------------------------------------------------------------------ config / history
    def get_config(self, key: str, default: Any = None) -> Any:
        with self.connect() as con:
            row = con.execute("SELECT value_json FROM app_config WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_config(self, key: str, value: Any) -> None:
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO app_config(key, value_json) VALUES (?, ?)", (key, json.dumps(value)))

    def log_analysis_run(self, dataset: str, summary: dict[str, Any]) -> None:
        with self.connect() as con:
            con.execute("INSERT INTO analysis_runs(ran_at, dataset, summary_json) VALUES (?,?,?)",
                        (datetime.now().isoformat(timespec="seconds"), self._check(dataset), json.dumps(summary, default=str)))

    def last_analysis_run(self, dataset: str | None = None) -> dict[str, Any] | None:
        q, p = "SELECT ran_at, dataset, summary_json FROM analysis_runs", ()
        if dataset:
            q, p = q + " WHERE dataset = ?", (dataset,)
        with self.connect() as con:
            row = con.execute(q + " ORDER BY id DESC LIMIT 1", p).fetchone()
        return {"ran_at": row[0], "dataset": row[1], **json.loads(row[2])} if row else None

    def save_forecast_meta(self, dataset: str, product_id: str, horizon: int, model: str, metrics: dict[str, float | None],
                           data_version: str) -> None:
        with self.connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO forecast_meta VALUES (?,?,?,?,?,?,?,?,?,?)",
                (self._check(dataset), product_id, int(horizon), model, metrics.get("mae"), metrics.get("wape"),
                 metrics.get("baseline_mae"), metrics.get("baseline_wape"),
                 datetime.now().isoformat(timespec="seconds"), data_version))

    def forecast_models_used(self, dataset: str) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query("SELECT * FROM forecast_meta WHERE dataset = ? ORDER BY trained_at DESC", con, params=(self._check(dataset),))

    # ------------------------------------------------------------------ maintenance
    def backup(self, dest: Path | str) -> Path:
        """Consistent online backup using SQLite's backup API."""
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = sqlite3.connect(self.db_path)
        try:
            dst = sqlite3.connect(dest)
            try:
                src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        return dest

    def reset_database(self) -> None:
        """Delete all stored data (user-initiated from Settings) and recreate the empty schema."""
        with self.connect() as con:
            for table in ("sales", "inventory", "purchases", "rejected_rows", "dataset_meta", "analysis_runs", "forecast_meta", "app_config"):
                con.execute(f"DELETE FROM {table}")
