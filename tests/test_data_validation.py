import pandas as pd
import pytest

from src.data import loader, validator as V
from src.data.repository import Repository
from tests.conftest import make_inventory, make_sales


def test_clean_file_has_no_errors():
    clean, rep = V.validate_sales(make_sales())
    assert len(clean) == 1 and rep.status in ("Clean", "Warnings") and not rep.has_errors


def test_empty_csv_is_reported_not_crashed():
    clean, rep = V.validate_sales(pd.DataFrame())
    assert clean.empty and rep.has_errors and any(i.code == "empty_file" for i in rep.issues)
    assert loader.ingest(pd.DataFrame(), None, None).is_empty


def test_empty_csv_file_reads_as_empty_frame(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    assert loader.read_csv(p).empty


def test_missing_required_column_is_fatal_and_listed():
    raw = make_sales().drop(columns=["quantity"])
    clean, rep = V.validate_sales(raw)
    assert clean.empty and any(i.code == "missing_required_quantity" for i in rep.issues)
    assert rep.rows_rejected == 1


def test_missing_optional_column_warns():
    clean, rep = V.validate_sales(make_sales().drop(columns=["category", "payment_method"]))
    assert len(clean) == 1 and any(i.code == "missing_category" for i in rep.issues)
    assert clean.iloc[0]["category"] == "Uncategorised"


def test_invalid_dates_are_rejected_with_reason_not_dropped_silently():
    raw = make_sales([{}, {"transaction_id": "T2", "transaction_date": "not a date"}, {"transaction_id": "T3", "transaction_date": ""}])
    clean, rep = V.validate_sales(raw)
    assert len(clean) == 1 and rep.rows_rejected == 2
    assert set(rep.rejected["reject_reason"]) == {"invalid transaction_date"}
    assert any(i.code == "invalid_date" and i.count == 2 for i in rep.issues)
    assert len(clean) + rep.rows_rejected == rep.rows_in


def test_day_first_dates_are_parsed():
    clean, _ = V.validate_sales(make_sales([{"transaction_date": "25/12/2025"}]))
    assert clean.iloc[0]["transaction_date"] == pd.Timestamp("2025-12-25")


def test_duplicate_transaction_detected_and_flagged():
    clean, rep = V.validate_sales(make_sales([{}, {}]))
    assert clean["is_duplicate"].tolist() == [False, True]
    assert any(i.code == "duplicate_transactions" for i in rep.issues)


def test_zero_negative_and_non_numeric_quantity():
    raw = make_sales([{"transaction_id": "A", "quantity": "0"}, {"transaction_id": "B", "quantity": "abc"},
                      {"transaction_id": "C", "quantity": "-2", "total_amount": "-200"}, {"transaction_id": "D"}])
    clean, rep = V.validate_sales(raw)
    assert set(clean["transaction_id"]) == {"C", "D"}
    assert clean.set_index("transaction_id").loc["C", "is_return"]          # negative quantity = return, kept
    assert rep.rows_rejected == 2


def test_invalid_and_negative_price_rejected():
    clean, rep = V.validate_sales(make_sales([{"transaction_id": "A", "unit_price": "-5"}, {"transaction_id": "B", "unit_price": "x"}]))
    assert clean.empty and rep.rows_rejected == 2


def test_rupee_strings_are_parsed():
    clean, _ = V.validate_sales(make_sales([{"unit_price": "₹1,250.50", "total_amount": "Rs. 2,501", "quantity": "2"}]))
    assert clean.iloc[0]["unit_price"] == 1250.5 and clean.iloc[0]["total_amount"] == 2501.0


def test_missing_cost_is_never_converted_to_zero():
    clean, rep = V.validate_sales(make_sales([{"unit_cost": ""}]))
    assert pd.isna(clean.iloc[0]["unit_cost"]) and clean.iloc[0]["cost_missing"]
    assert any(i.code == "missing_unit_cost" for i in rep.issues)


def test_inconsistent_total_flagged_and_recorded_total_kept():
    clean, rep = V.validate_sales(make_sales([{"total_amount": "150"}]))
    assert clean.iloc[0]["total_mismatch"] and clean.iloc[0]["total_amount"] == 150
    assert any(i.code == "inconsistent_total" for i in rep.issues)


def test_missing_total_is_computed_and_flagged():
    clean, _ = V.validate_sales(make_sales([{"total_amount": "", "discount": "10"}]))
    assert clean.iloc[0]["total_amount"] == pytest.approx(180) and clean.iloc[0]["total_computed"]


def test_discount_out_of_range_rejected():
    clean, rep = V.validate_sales(make_sales([{"discount": "150"}]))
    assert clean.empty and rep.rows_rejected == 1


def test_discount_amount_mode_converts_to_percent():
    clean, _ = V.validate_sales(make_sales([{"discount": "20", "total_amount": "180"}]), discount_mode="amount")
    assert clean.iloc[0]["discount"] == pytest.approx(10)


def test_negative_stock_warning_and_empty_inventory():
    inv, rep = V.validate_inventory(make_inventory([{"current_stock": "-3"}]))
    assert inv.iloc[0]["negative_stock"] and any(i.code == "negative_stock" for i in rep.issues)
    inv, rep = V.validate_inventory(pd.DataFrame())
    assert inv.empty and rep.has_errors


def test_inventory_duplicate_product_keeps_last_and_reports():
    inv, rep = V.validate_inventory(make_inventory([{"current_stock": "5"}, {"current_stock": "9"}]))
    assert len(inv) == 1 and inv.iloc[0]["current_stock"] == 9 and any(i.code == "duplicate_product_id" for i in rep.issues)


def test_column_mapping_synonyms_and_fuzzy():
    m = V.suggest_mapping(["Date", "SKU", "Item Name", "Qty", "Price", "Total", "Discount %", "Cost Price"], V.SALES_SPEC)
    assert m["transaction_date"] == "Date" and m["product_id"] == "SKU" and m["quantity"] == "Qty"
    assert m["unit_price"] == "Price" and m["total_amount"] == "Total" and m["unit_cost"] == "Cost Price"


def test_ingest_with_custom_headers_end_to_end():
    raw = pd.DataFrame({"Bill Date": ["2026-02-01"], "SKU": ["X1"], "Qty": ["3"], "Rate": ["50"], "Amount": ["150"]})
    ds = loader.ingest(raw, None, None)
    assert len(ds.sales) == 1 and ds.sales.iloc[0]["product_id"] == "X1" and ds.sales.iloc[0]["total_amount"] == 150


def test_cost_estimated_from_inventory_and_flagged():
    sales = make_sales([{"unit_cost": ""}])
    ds = loader.ingest(sales, make_inventory([{"unit_cost": "70"}]), None)
    row = ds.sales.iloc[0]
    assert row["cost_source"] == "inventory_estimate" and row["unit_cost_eff"] == 70


def test_cross_validation_reports_unknown_products():
    ds = loader.ingest(make_sales([{"product_id": "PX"}]), make_inventory(), None)
    assert any(i.code == "sales_products_not_in_inventory" for i in ds.cross_issues)


def test_repository_roundtrip_and_idempotent_demo(tmp_path, demo_ds):
    repo = Repository(tmp_path / "t.db")
    repo.ensure_demo_loaded()
    repo.ensure_demo_loaded()                       # no duplicate inserts on repeat
    back = repo.load_dataset("demo")
    assert len(back.sales) == len(demo_ds.sales) and len(back.inventory) == len(demo_ds.inventory)
    assert repo.get_active() == "demo"
    repo.save_dataset(loader.ingest(make_sales(), make_inventory(), None), "uploaded")
    repo.set_active("uploaded")
    assert repo.get_active() == "uploaded" and len(repo.load_dataset("demo").sales) == len(demo_ds.sales)   # data kept separate
    repo.delete_dataset("uploaded")
    assert repo.get_active() == "demo"
    repo.backup(tmp_path / "b.db")
    assert (tmp_path / "b.db").stat().st_size > 0


def test_messy_real_world_export_end_to_end(tmp_path):
    """Indian-style export: odd headers, DD/MM/YYYY dates, rupee strings, blanks, junk rows, a duplicate bill and a refund."""
    csv = tmp_path / "messy.csv"
    csv.write_text(
        "Bill No,Bill Date,Item Code,Item,Qty,Rate,Disc %,Amount,Cost Price\n"
        "B1,05/01/2026,A1,Atta 5kg,2,\"₹1,200.00\",0,\"₹2,400.00\",1000\n"
        "B2,06/01/2026,A1,Atta 5kg,1,Rs. 1200,10,\"1,080\",1000\n"
        "B2,06/01/2026,A1,Atta 5kg,1,Rs. 1200,10,\"1,080\",1000\n"          # duplicate bill line
        "B3,31/02/2026,A1,Atta 5kg,1,1200,0,1200,1000\n"                  # impossible date
        "B4,07/01/2026,A2,Dal 1kg,abc,150,0,150,\n"                       # junk quantity
        "B5,08/01/2026,A2,Dal 1kg,3,150,0,450,\n"                         # missing cost
        "B6,09/01/2026,A2,Dal 1kg,-1,150,0,-150,120\n",                   # refund
        encoding="utf-8",
    )
    ds = loader.ingest(loader.read_csv(csv), None, None)
    rep = ds.reports["sales"]
    assert rep.rows_in == 7 and rep.rows_valid == 5 and rep.rows_rejected == 2          # nothing silently lost
    assert set(rep.rejected["reject_reason"]) == {"invalid transaction_date", "invalid quantity"}
    assert ds.sales.loc[ds.sales["transaction_id"] == "B1", "transaction_date"].iloc[0] == pd.Timestamp("2026-01-05")   # day-first
    assert ds.sales.loc[ds.sales["transaction_id"] == "B1", "unit_price"].iloc[0] == 1200
    assert ds.sales["is_duplicate"].sum() == 1 and ds.sales["is_return"].sum() == 1
    assert ds.sales.loc[ds.sales["transaction_id"] == "B5", "cost_missing"].iloc[0]
    codes = {i.code for i in rep.issues}
    assert {"invalid_date", "invalid_quantity", "duplicate_transactions", "missing_unit_cost", "returns_present"} <= codes
    from src.analytics import profitability as P
    s = P.compute_summary(ds.sales)
    assert s.revenue == pytest.approx(2400 + 1080 + 450 - 150)             # duplicate excluded, refund netted
    assert s.cogs == pytest.approx(2 * 1000 + 1 * 1000 + (-1) * 120)       # B5 has no cost: excluded, not zero-costed
    assert s.cost_coverage_pct < 100
