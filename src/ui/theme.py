"""Visual identity: deep navy surface, teal brand accent, a validated colour-blind-safe chart palette."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config

# Categorical chart palette: validated for the dark surface (contrast >= 3:1, adjacent CVD delta-E >= 8).
# Colour follows the entity: revenue is always blue, gross profit always aqua, costs orange.
PALETTE = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
C_REVENUE, C_COST, C_PROFIT, C_WARN, C_ACCENT = PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3], "#14b8a6"
INK, INK_2, INK_MUTED = "#e6edf7", "#aab6cc", "#8593ad"
SURFACE, CARD, BORDER = "#0b1220", "#111a2e", "#22304d"
SEVERITY_STYLE = {"High": ("#d03b3b", "▲ High"), "Medium": ("#ec835a", "◆ Medium"), "Low": ("#8593ad", "● Low")}

_CSS = f"""
<style>
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; }}
[data-testid="stMetricLabel"] p {{ color: {INK_2}; font-size: 0.82rem; }}
[data-testid="stMetricValue"] {{ font-size: 1.55rem; font-weight: 650; }}
[data-testid="stSidebar"] {{ border-right: 1px solid {BORDER}; }}
.vy-hero {{ border-left: 4px solid {C_ACCENT}; padding: 4px 0 4px 14px; margin-bottom: 14px; }}
.vy-hero h1 {{ margin: 0; font-size: 1.9rem; }}
.vy-hero p {{ margin: 2px 0 0 0; color: {INK_2}; }}
.vy-badge {{ display:inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.78rem; font-weight: 600; border: 1px solid {BORDER}; }}
.vy-card {{ background: {CARD}; border: 1px solid {BORDER}; border-radius: 12px; padding: 14px 16px; margin-bottom: 10px; }}
.vy-muted {{ color: {INK_MUTED}; font-size: 0.85rem; }}
.vy-sev {{ font-weight: 700; }}
div[data-testid="stTabs"] button[role="tab"] {{ font-weight: 600; }}
</style>
"""


def apply_theme() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(f'<div class="vy-hero"><h1>{title}</h1>{sub}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = C_ACCENT) -> str:
    return f'<span class="vy-badge" style="color:{color};border-color:{color}">{text}</span>'


def severity_label(sev: str) -> str:
    color, label = SEVERITY_STYLE.get(sev, SEVERITY_STYLE["Low"])
    return f'<span class="vy-sev" style="color:{color}">{label}</span>'


def scale_money(values: pd.Series) -> tuple[pd.Series, str, str]:
    """Scale rupee values to crore / lakh / rupees for readable axes. Returns (scaled, unit_label, hover_suffix)."""
    top = float(values.abs().max()) if len(values) else 0.0
    if top >= 1e7:
        return values / 1e7, "₹ crore", " Cr"
    if top >= 1e5:
        return values / 1e5, "₹ lakh", " L"
    return values, "₹", ""


def style_fig(fig: go.Figure, height: int = 360, legend: bool = True, yprefix: str = "") -> go.Figure:
    fig.update_layout(
        template="plotly_dark", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=INK_2, size=12), colorway=PALETTE, margin=dict(l=8, r=8, t=70, b=8),
        hoverlabel=dict(bgcolor=CARD, font_color=INK, bordercolor=BORDER), hovermode="x unified" if legend else "closest",
        showlegend=legend, legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0, font=dict(color=INK_2)),
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left", y=0.97, yanchor="top"),
    )
    fig.update_xaxes(showgrid=False, linecolor=BORDER, tickfont=dict(color=INK_MUTED), zeroline=False)
    fig.update_yaxes(gridcolor="rgba(148,163,184,0.14)", zeroline=False, tickfont=dict(color=INK_MUTED), tickprefix=yprefix)
    return fig


def empty_state(message: str) -> None:
    st.markdown(f'<div class="vy-card vy-muted">{message}</div>', unsafe_allow_html=True)


def brand_caption() -> str:
    return f"{config.APP_NAME} · {config.TAGLINE}"
