"""BioStreamer analyst UI — Streamlit panel over the unified FastAPI layer.

Three views:
  1. Fleet overview   -- warehouse KPIs and the substrate x concentration
                          matrix, reproducing the source paper's Table 2.
  2. Reactor explorer -- cumulative-yield curves and pH/VFA traces for any
                          reactor, with the reference-design cohort flagged.
  3. Research chat    -- the hybrid RAG endpoint, with its retrieval trace
                          exposed so an answer can be audited against the
                          passages and warehouse numbers that produced it.

This file talks to the FastAPI service exclusively over HTTP; it holds no
database or vector-store client of its own, so it can be pointed at a remote
deployment of the API by changing BIOSTREAM_API_URL alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from src.common import science as sci
from src.common.config import get_settings

st.set_page_config(page_title="BioStreamer", page_icon="🧫", layout="wide")

API_URL = get_settings().api_base_url


# --------------------------------------------------------------------------- #
# API client helpers
# --------------------------------------------------------------------------- #

@st.cache_data(ttl=30)
def api_get(path: str, params: dict | None = None) -> dict | list:
    resp = requests.get(f"{API_URL}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def api_post(path: str, json: dict) -> dict:
    resp = requests.post(f"{API_URL}{path}", json=json, timeout=60)
    resp.raise_for_status()
    return resp.json()


def api_health() -> dict | None:
    try:
        return api_get("/health")
    except requests.RequestException:
        return None


# --------------------------------------------------------------------------- #
# Sidebar: connection status
# --------------------------------------------------------------------------- #

st.sidebar.title("🧫 BioStreamer")
st.sidebar.caption(
    "Grounded in [DOI 10.30574/gscbps.2024.29.2.0423]"
    "(https://doi.org/10.30574/gscbps.2024.29.2.0423)"
)

health = api_health()
if health is None:
    st.sidebar.error(f"API unreachable at {API_URL}")
    st.error(
        f"Cannot reach the BioStreamer API at `{API_URL}`. Start it with:\n\n"
        "```\npython -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000\n```"
    )
    st.stop()

status_icon = "🟢" if health["status"] == "ok" else "🟡"
st.sidebar.markdown(f"{status_icon} API status: **{health['status']}**")
for check, ok in health["checks"].items():
    st.sidebar.markdown(f"{'✅' if ok else '❌'} {check}")
if not health["checks"]["llm"]:
    st.sidebar.info(
        "LLM not configured — chat runs in retrieval-only mode "
        "(literature + warehouse context, no synthesized answer)."
    )

page = st.sidebar.radio("View", ["Fleet overview", "Reactor explorer", "Research chat"])


# --------------------------------------------------------------------------- #
# View 1: Fleet overview
# --------------------------------------------------------------------------- #

def render_fleet_overview() -> None:
    st.title("Fleet overview")
    st.caption("100 parallel bioreactor lines — 36 reproducing the published design, 64 exploratory.")

    summary = api_get("/stats/summary")
    totals = summary["warehouse_totals"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reactors", totals["reactor_count"])
    c2.metric("Avg. final yield", f"{totals['avg_final_yield_ml']} ml")
    c3.metric("Worst min. pH observed", totals["worst_min_ph"])
    c4.metric("Soured reactor-days", totals["total_soured_reactor_days"])

    st.divider()
    st.subheader("Reference cohort vs. published Table 2")
    st.caption(
        "Mean daily biogas yield (ml/day) — the platform's 36 reference reactors "
        "replicate the paper's RCBD exactly; these numbers should match Table 2."
    )

    matrix = pd.DataFrame(summary["reference_yield_matrix"])
    published = []
    for _, row in matrix.iterrows():
        published.append(sci.MEAN_DAILY_YIELD_ML[(row["substrate_code"], row["slurry_level_code"])])
    matrix["published_ml_day"] = published
    matrix["observed_mean_ml_day"] = matrix["observed_mean_ml_day"].astype(float)
    matrix["delta"] = (matrix["observed_mean_ml_day"] - matrix["published_ml_day"]).round(3)

    pivot_obs = matrix.pivot(index="substrate_code", columns="slurry_level_code", values="observed_mean_ml_day")
    pivot_pub = matrix.pivot(index="substrate_code", columns="slurry_level_code", values="published_ml_day")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Platform (simulated)**")
        st.dataframe(pivot_obs.style.format("{:.3f}").background_gradient(cmap="Greens"), use_container_width=True)
    with col_b:
        st.markdown("**Published (paper, Table 2)**")
        st.dataframe(pivot_pub.style.format("{:.2f}").background_gradient(cmap="Greens"), use_container_width=True)

    max_delta = matrix["delta"].abs().max()
    st.caption(f"Maximum absolute deviation from published means: {max_delta:.3f} ml/day.")

    st.divider()
    st.subheader("Performance by substrate")
    perf = pd.DataFrame(summary_by_substrate := api_get("/reactors", {"limit": 100}))
    perf_agg = (
        pd.DataFrame(perf)
        .groupby("substrate_code")
        .agg(
            reactors=("reactor_id", "count"),
            avg_final_yield_ml=("final_cumulative_ml", "mean"),
            avg_ph=("mean_ph", "mean"),
            soured_days=("soured_days", "sum"),
        )
        .reset_index()
        .sort_values("avg_final_yield_ml", ascending=False)
    )
    fig = go.Figure(
        go.Bar(
            x=perf_agg["substrate_code"],
            y=perf_agg["avg_final_yield_ml"],
            marker_color=["#e07a3e" if s == "BLEND" else "#3e8ee0" for s in perf_agg["substrate_code"]],
            text=perf_agg["avg_final_yield_ml"].round(1),
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis_title="Mean final cumulative yield (ml)",
        xaxis_title="Substrate",
        height=380,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "BLEND = exploratory reactors spanning the continuous bean-fraction x "
        "5-25%TS design space, beyond the four discrete substrates the paper tested."
    )


# --------------------------------------------------------------------------- #
# View 2: Reactor explorer
# --------------------------------------------------------------------------- #

def render_reactor_explorer() -> None:
    st.title("Reactor explorer")

    reactors = pd.DataFrame(api_get("/reactors", {"limit": 100}))
    reactors = reactors.sort_values("reactor_id")

    col1, col2 = st.columns([1, 3])
    with col1:
        substrate_filter = st.multiselect(
            "Substrate", sorted(reactors["substrate_code"].unique()), default=None
        )
        ref_only = st.checkbox("Reference design only", value=False)

    view = reactors
    if substrate_filter:
        view = view[view["substrate_code"].isin(substrate_filter)]
    if ref_only:
        view = view[view["is_reference_design"]]

    with col1:
        reactor_id = st.selectbox("Reactor", view["reactor_id"].tolist())

    reactor = view[view["reactor_id"] == reactor_id].iloc[0]

    with col2:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Substrate", reactor["substrate_code"])
        m2.metric("Slurry %TS", f"{reactor['slurry_concentration_pct_ts']:.1f}%")
        m3.metric("Final yield", f"{reactor['final_cumulative_ml']:.1f} ml")
        m4.metric("Soured days", int(reactor["soured_days"] or 0))
        if reactor["beyond_published_envelope"]:
            st.warning(
                f"{reactor_id} runs at {reactor['slurry_concentration_pct_ts']:.1f}% TS, above the "
                f"published envelope ceiling of {sci.PUBLISHED_TS_CEILING_PCT}% TS — this is "
                "extrapolation beyond the source study's design space."
            )
        elif reactor["is_reference_design"]:
            st.success(f"{reactor_id} replicates the published RCBD ({reactor['substrate_ratio']} @ {reactor['slurry_ratio']}).")

    telemetry = pd.DataFrame(api_get(f"/telemetry/{reactor_id}"))

    tab1, tab2 = st.tabs(["Yield curve", "pH & process state"])

    with tab1:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=telemetry["day_index"], y=telemetry["cumulative_biogas_yield_ml"],
                mode="lines+markers", name="Cumulative yield (ml)", line=dict(color="#3e8ee0"),
            )
        )
        fig.add_trace(
            go.Bar(
                x=telemetry["day_index"], y=telemetry["daily_biogas_ml"],
                name="Daily yield (ml)", marker_color="rgba(62,142,224,0.3)", yaxis="y2",
            )
        )
        fig.update_layout(
            xaxis_title="Day",
            yaxis=dict(title="Cumulative yield (ml)"),
            yaxis2=dict(title="Daily yield (ml)", overlaying="y", side="right"),
            height=420, margin=dict(t=20, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        state_colors = {
            "LAG": "#9e9e9e", "ACIDOGENIC": "#e0a83e", "METHANOGENIC": "#3ee06a",
            "SOURED": "#e0473e", "RECOVERING": "#3ec8e0",
        }
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=telemetry["day_index"], y=telemetry["daily_ph"],
                mode="lines+markers", name="Daily pH",
                marker=dict(color=[state_colors.get(s, "#888") for s in telemetry["process_state"]], size=8),
                line=dict(color="#888", width=1),
            )
        )
        fig.add_hline(y=6.5, line_dash="dot", line_color="orange", annotation_text="inhibition onset (pH 6.5)")
        fig.add_hline(y=6.0, line_dash="dot", line_color="red", annotation_text="souring threshold (pH 6.0)")
        fig.update_layout(xaxis_title="Day", yaxis_title="pH", height=420, margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

        legend_md = "  ".join(f"<span style='color:{c}'>●</span> {s}" for s, c in state_colors.items())
        st.markdown(legend_md, unsafe_allow_html=True)

    with st.expander("Raw telemetry"):
        st.dataframe(telemetry, use_container_width=True, hide_index=True)


# --------------------------------------------------------------------------- #
# View 3: Research chat
# --------------------------------------------------------------------------- #

def render_research_chat() -> None:
    st.title("Research chat")
    st.caption(
        "Hybrid retrieval: every question queries both the literature index and "
        "the live telemetry warehouse. Expand the retrieval trace below any "
        "answer to see exactly what grounded it."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn["answer"]:
                st.write(turn["answer"])
            else:
                st.info(turn["note"] or "No answer synthesized.")
            with st.expander(f"Retrieval trace ({len(turn['literature'])} passages)"):
                for hit in turn["literature"]:
                    badge = "📄 published" if hit["provenance"] == "published_finding" else "🧪 domain context"
                    st.markdown(f"**{badge}** · `{hit['section']}` · score={hit['score']}")
                    st.caption(hit["text"])
                st.markdown("**Warehouse aggregates supplied:**")
                st.json(turn["sql_context"])

    question = st.chat_input("Ask about substrates, yields, pH, or process failures...")
    if question:
        with st.spinner("Retrieving context and generating an answer..."):
            try:
                result = api_post("/chat", {"question": question, "top_k": 5})
            except requests.RequestException as exc:
                st.error(f"Request failed: {exc}")
                return

        st.session_state.chat_history.append(
            {
                "question": question,
                "answer": result.get("answer"),
                "note": result.get("note"),
                "literature": result.get("literature_context", []),
                "sql_context": result.get("sql_context", {}),
            }
        )
        st.rerun()

    with st.sidebar:
        st.divider()
        if st.button("Clear chat history"):
            st.session_state.chat_history = []
            st.rerun()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

if page == "Fleet overview":
    render_fleet_overview()
elif page == "Reactor explorer":
    render_reactor_explorer()
else:
    render_research_chat()
