"""VyaparAI entry point: branding, navigation and the 8 sections.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

import config  # noqa: E402

st.set_page_config(page_title=f"{config.APP_NAME} · AI Business Copilot", page_icon="📈", layout="wide",
                   initial_sidebar_state="expanded")
st.logo(str(ROOT / "assets" / "logo.svg"), size="large")

PAGES = [
    st.Page("pages/1_Overview.py", title="Overview", icon="🏠", default=True),
    st.Page("pages/2_Sales_Analytics.py", title="Sales Analytics", icon="📊"),
    st.Page("pages/3_Inventory_Intelligence.py", title="Inventory Intelligence", icon="📦"),
    st.Page("pages/4_Demand_Forecasting.py", title="Demand Forecasting", icon="🔮"),
    st.Page("pages/5_Profit_Leakage.py", title="Profit Leakage Detection", icon="🕵️"),
    st.Page("pages/6_AI_Business_Analyst.py", title="AI Business Analyst", icon="🤖"),
    st.Page("pages/7_Data_Management.py", title="Data Management", icon="🗂️"),
    st.Page("pages/8_Settings.py", title="Settings and Model Status", icon="⚙️"),
]

st.navigation(PAGES).run()
