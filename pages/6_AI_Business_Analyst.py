"""AI Business Analyst: grounded Q&A (local free LLM when available, labelled fallback otherwise)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

import config  # noqa: E402
from src.genai.analyst import BusinessAnalyst, HELP_TEXT  # noqa: E402
from src.ui import components as C  # noqa: E402
from src.ui import state, theme  # noqa: E402

theme.apply_theme()
ctx = state.ensure_ready()
C.sidebar_status(ctx)
theme.page_header("AI Business Analyst", "Ask questions in plain language. Numbers always come from calculated data, never from the model's memory.")
C.demo_banner(ctx)

mode, detail = state.llm_mode_summary()
is_ai = mode.startswith("AI")
st.markdown(theme.badge(mode, theme.C_PROFIT if is_ai else theme.C_WARN), unsafe_allow_html=True)
if not is_ai:
    st.warning(f"**Fallback mode.** {detail} Answers are built from templates using the same calculated data, and are labelled as such. "
               f"For conversational answers, install Ollama (free) and run `ollama pull {config.OLLAMA_MODEL}`; see the README.", icon="🟠")
else:
    st.caption(detail)

if ctx.ds.is_empty:
    theme.empty_state("No valid sales data is loaded. Go to <b>Data Management</b>.")
    st.stop()

SUGGESTIONS = ["Which products are losing me money?", "What should I reorder this week?", "Why did gross profit decrease?",
               "Which products have the highest inventory risk?", "Which categories are performing best?",
               "What are my biggest potential profit leaks?", "Which products are selling more slowly than before?",
               "Summarize my business performance for the last 30 days.", "Recommend five actions to improve inventory efficiency."]

if "chat" not in st.session_state:
    st.session_state["chat"] = []
chat: list[dict] = st.session_state["chat"]

with st.expander("Example questions", expanded=not chat):
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTIONS):
        if cols[i % 2].button(q, key=f"sugg_{i}", width="stretch"):
            st.session_state["pending_q"] = q
    st.caption("You can also ask for a forecast: *Forecast demand for Amul Gold Milk 1L for the next 7 days.*")

for m in chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant":
            tag = {"llm": f"AI answer · {m.get('backend', '')}", "fallback": "Fallback answer (templates, no language model)", "limitation": "Cannot answer"}[m["mode"]]
            st.caption(tag)
            for w in m.get("warnings", []):
                st.warning(w, icon="⚠️")
            if m.get("facts"):
                with st.expander("Calculated data used (source of truth)"):
                    st.caption("Tools run: " + ", ".join(f"`{t['tool']}`" for t in m["tools"] if t["ok"]) + ". Read-only; nothing was changed.")
                    st.json(m["facts"], expanded=False)

question = st.chat_input("Ask about sales, profit, stock, forecasts...", max_chars=500)
question = question or st.session_state.pop("pending_q", None)
if question:
    chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    analyst = BusinessAnalyst(ctx.analysis, state.get_client())
    with st.chat_message("assistant"):
        with st.spinner("Calculating from your data..."):
            ans = analyst.ask(question, [{"role": m["role"], "content": m["content"]} for m in chat[:-1]])
        st.markdown(ans.text)
    chat.append({"role": "assistant", "content": ans.text, "mode": ans.mode, "backend": ans.backend, "warnings": ans.warnings,
                 "facts": json.loads(json.dumps(ans.facts, default=str)), "tools": ans.tools})
    st.rerun()

c1, c2 = st.columns([1, 5])
if chat and c1.button("Clear chat", key="chat_clear"):
    st.session_state["chat"] = []
    st.rerun()
st.caption("The analyst can only run a fixed set of read-only analysis tools. It cannot run SQL, code or commands, change inventory, or place orders. "
           "Anomalies are items to review, not proof of wrongdoing. Gross profit is not net profit.")
with st.expander("What can I ask?"):
    st.text(HELP_TEXT)
