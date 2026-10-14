"""Data management: upload, map columns, validate, load demo data, reset."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.data import loader  # noqa: E402
from src.data import validator as V  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("Data Management", "Load the demo data or upload your own CSV files. Problem rows are reported, never silently dropped.")

info = ctx.repo.dataset_info()
c1, c2 = st.columns(2)
with c1:
    st.markdown("#### Demo data (synthetic)")
    st.caption("Fictional Indian retailer, ~12 months of daily sales, 75 products, with deliberately inserted anomalies. Amounts in ₹. Not real business results.")
    if "demo" in info:
        st.caption(f"Loaded {info['demo']['loaded_at']} · {info['demo']['sales_rows']:,} sales lines")
    b1, b2 = st.columns(2)
    if b1.button("Reset to demo data", key="dm_reset_demo", help="Reloads the demo CSVs and makes them active. Your uploaded data is kept."):
        ctx.repo.reset_demo()
        st.cache_resource.clear()
        st.rerun()
    if "uploaded" in info and ctx.active != "demo":
        if b2.button("Switch to demo", key="dm_to_demo"):
            ctx.repo.set_active("demo")
            st.rerun()
with c2:
    st.markdown("#### Your uploaded data")
    if "uploaded" in info:
        st.caption(f"Loaded {info['uploaded']['loaded_at']} · {info['uploaded']['sales_rows']:,} sales lines")
        b1, b2 = st.columns(2)
        if ctx.active != "uploaded" and b1.button("Use uploaded data", key="dm_to_up"):
            ctx.repo.set_active("uploaded")
            st.rerun()
        if b2.button("Delete uploaded data", key="dm_del_up"):
            st.session_state["confirm_del"] = True
        if st.session_state.get("confirm_del"):
            st.warning("This permanently deletes your uploaded data from the local database. Your original CSV files are not touched.")
            y, n = st.columns(2)
            if y.button("Yes, delete", key="dm_del_yes", type="primary"):
                was_active = ctx.active == "uploaded"
                ctx.repo.delete_dataset("uploaded")
                if was_active:
                    ctx.repo.set_active("demo")
                st.session_state["confirm_del"] = False
                st.cache_resource.clear()
                st.rerun()
            if n.button("Cancel", key="dm_del_no"):
                st.session_state["confirm_del"] = False
                st.rerun()
    else:
        st.caption("Nothing uploaded yet.")
st.markdown(f"**Active dataset:** {'Demo (synthetic)' if ctx.active == 'demo' else 'Your uploaded data'}")

tab_up, tab_quality, tab_sample, tab_schema = st.tabs(["Upload CSV files", "Data-quality report", "Sample files", "Expected columns"])

with tab_up:
    st.write("Upload a **sales** file (required) and optionally **inventory** and **purchases** files. Column names are matched automatically; you can correct the mapping before loading.")
    f_sales = st.file_uploader("Sales transactions (CSV)", type=["csv"], key="up_sales")
    f_inv = st.file_uploader("Inventory (CSV, optional)", type=["csv"], key="up_inv")
    f_pur = st.file_uploader("Purchases / restocking (CSV, optional)", type=["csv"], key="up_pur")

    raws: dict[str, pd.DataFrame] = {}
    for kind, f in (("sales", f_sales), ("inventory", f_inv), ("purchases", f_pur)):
        if f is not None:
            try:
                raws[kind] = loader.read_csv(f)
            except Exception as e:  # malformed CSV
                st.error(f"Could not read the {kind} file: {type(e).__name__}. Check that it is a valid CSV.")

    mappings: dict[str, dict[str, str | None]] = {}
    for kind, raw in raws.items():
        st.markdown(f"##### {kind.capitalize()} file preview ({len(raw):,} rows, {len(raw.columns)} columns)")
        if raw.empty:
            st.error(f"The {kind} file is empty.")
            continue
        st.dataframe(raw.head(8), hide_index=True, width="stretch")
        suggested = V.suggest_mapping(raw.columns, V.SPECS[kind])
        with st.expander(f"Column mapping for {kind}", expanded=any(sp.required and not suggested[sp.name] for sp in V.SPECS[kind])):
            options = ["(not available)"] + list(raw.columns)
            cols = st.columns(3)
            m: dict[str, str | None] = {}
            for i, sp in enumerate(V.SPECS[kind]):
                default = suggested[sp.name]
                idx = options.index(default) if default in options else 0
                choice = cols[i % 3].selectbox(f"{sp.name}{' *' if sp.required else ''}", options, index=idx, help=sp.description, key=f"map_{kind}_{sp.name}")
                m[sp.name] = None if choice == "(not available)" else choice
            mappings[kind] = m
    discount_mode = "percent"
    if "sales" in raws:
        discount_mode = st.radio("Your discount column contains", ["percent", "amount"], horizontal=True,
                                 format_func=lambda x: "Percentages (e.g. 10 = 10%)" if x == "percent" else "Rupee amounts per line", key="up_disc")

    if "sales" in raws and not raws["sales"].empty:
        if st.button("Validate and load", type="primary", key="up_go"):
            with st.spinner("Validating and saving..."):
                ds = loader.ingest(raws.get("sales"), raws.get("inventory"), raws.get("purchases"), mappings, discount_mode, source=loader.UPLOADED_LABEL)
                if ds.is_empty:
                    st.session_state["last_report"] = ds
                    st.error("No valid sales rows remain after validation, so nothing was loaded. See the report below.")
                else:
                    ctx.repo.save_dataset(ds, loader.UPLOADED_LABEL)
                    ctx.repo.set_active(loader.UPLOADED_LABEL)
                    st.session_state["last_report"] = ds
                    st.session_state.pop("use_ml_demand", None)
                    st.cache_resource.clear()
                    st.success(f"Loaded {len(ds.sales):,} sales lines. Your data is now the active dataset. Review the report below.")
    elif not raws:
        st.info("Choose at least a sales CSV to begin. The sample files in the next tab show the expected layout.")

    rep_ds = st.session_state.get("last_report")
    if rep_ds is not None:
        st.markdown("##### Validation report for this upload")
        for kind, rep in rep_ds.reports.items():
            if rep.rows_in or rep.issues:
                st.markdown(f"**{kind.capitalize()}**: {rep.rows_in:,} rows in · {rep.rows_valid:,} valid · {rep.rows_rejected:,} set aside · status **{rep.status}**")
                st.dataframe(rep.to_frame(), hide_index=True, width="stretch")
                if rep.rows_rejected:
                    st.markdown("Rows set aside (not analysed). Fix them in your file and upload again:")
                    st.dataframe(rep.rejected.head(200), hide_index=True, width="stretch")
                    C.csv_button(rep.rejected, f"vyaparai_rejected_{kind}.csv", f"Download rejected {kind} rows", key=f"rej_{kind}")

with tab_quality:
    st.markdown("Report for the **active** dataset. Rows are never silently discarded: rows that cannot be analysed are listed, suspicious ones are kept and flagged.")
    st.markdown(f"Overall status: **{ctx.ds.quality_status()}**")
    for kind, rep in ctx.ds.reports.items():
        st.markdown(f"**{kind.capitalize()}**: {rep.rows_in:,} rows in · {rep.rows_valid:,} valid · {rep.rows_rejected:,} set aside")
        if rep.issues:
            st.dataframe(rep.to_frame(), hide_index=True, width="stretch")
        else:
            st.success("No issues found." if rep.rows_in else "No file loaded for this section.")
        if rep.rows_rejected:
            st.dataframe(rep.rejected.head(200), hide_index=True, width="stretch")
    for iss in ctx.ds.cross_issues:
        (st.warning if iss.severity == "warning" else st.info)(f"{iss.message} {iss.impact}")
    st.markdown("**How issues affect the analysis**")
    st.markdown("- *Duplicates* are excluded from revenue and profit.\n- *Missing costs* are estimated from the inventory file when possible (flagged); otherwise those lines are left out of margin figures, never treated as zero cost.\n"
                "- *Inconsistent totals* keep the recorded amount in revenue and are listed in Profit Leakage Detection.\n- *Returns* (negative quantity) reduce revenue and cost of goods sold.\n"
                "- *Invalid dates, quantities or prices* are set aside and excluded.")

with tab_sample:
    st.write("Download the demo files to see the expected layout, or use them as a template for your own data. All values are synthetic.")
    files = loader.demo_csv_bytes()
    cols = st.columns(3)
    for col, (name, data) in zip(cols, files.items()):
        col.download_button(name, data, file_name=name, mime="text/csv", key=f"dl_{name}")
    st.caption("`discount` is a percentage (0-100). `purchase_cost` is the total cost of the purchase line. Amounts are in ₹. See `docs/DATA_DICTIONARY.md` for every column.")

with tab_schema:
    for kind, spec in V.SPECS.items():
        st.markdown(f"**{kind.capitalize()}**")
        st.dataframe(pd.DataFrame([{"Column": sp.name, "Required": "Yes" if sp.required else "No", "Type": {"str": "text", "date": "date", "num": "number"}[sp.kind],
                                    "Description": sp.description, "If missing": sp.fallback_note or ("Cannot analyse" if sp.required else "")} for sp in spec]),
                     hide_index=True, width="stretch")
