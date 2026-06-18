from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agents.orchestrator import OrchestratorAgent
from app.data_store import DataStore


st.set_page_config(
    page_title="Telecom Multi-Agent RCA",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
    [data-testid="stMetric"] {
      border: 1px solid #d7dde6;
      border-radius: 8px;
      padding: 0.65rem 0.8rem;
      background: #ffffff;
    }
    div[data-testid="stTabs"] button { font-weight: 600; }
    .small-label { color: #5b6472; font-size: 0.86rem; margin-bottom: 0.1rem; }
    .trace-summary { color: #253041; font-size: 0.95rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_store() -> DataStore:
    return DataStore(ROOT / "data")


@st.cache_data(show_spinner=False)
def run_report(incident_id: str) -> dict:
    store = load_store()
    return OrchestratorAgent(store).run(incident_id).to_dict()


def _kpi_table(kpis: dict) -> pd.DataFrame:
    rows = []
    for metric, payload in kpis.items():
        normal = payload.get("normal_range", ["", ""])
        rows.append(
            {
                "Metric": metric,
                "Latest": payload.get("latest"),
                "Unit": payload.get("unit", ""),
                "Normal": f"{normal[0]}-{normal[1]}" if len(normal) == 2 else "",
            }
        )
    return pd.DataFrame(rows)


store = load_store()
incidents = store.list_incidents()
incident_options = {
    f"{incident['incident_id']} | {incident['domain']} | {incident['symptom']}": incident["incident_id"]
    for incident in incidents
}

st.sidebar.title("Telecom RCA")
selected_label = st.sidebar.selectbox("Incident", list(incident_options))
incident_id = incident_options[selected_label]
incident = store.get_incident(incident_id)
report = run_report(incident_id)

st.title("Multi-Agent Telecom RCA")
st.caption("Tool-assisted, SOP-guided, consensus-verified troubleshooting prototype")

left, mid, right, last = st.columns(4)
left.metric("Domain", report["domain"])
mid.metric("Severity", report["severity"])
right.metric("Confidence", f"{report['confidence']:.2f}")
last.metric("Agent steps", report["metrics"]["agent_steps"])

tab_overview, tab_trace, tab_evidence, tab_remediation = st.tabs(
    ["Incident Overview", "Agent Trace", "Evidence & Root Cause", "Remediation & Validation"]
)

with tab_overview:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.subheader("Incident")
        st.write(
            {
                "incident_id": incident["incident_id"],
                "symptom": incident["symptom"],
                "description": incident["description"],
                "primary_ne": incident["primary_ne"],
                "service_impact": incident["service_impact"],
                "affected_services": incident["affected_services"],
            }
        )
        st.subheader("Topology Edges")
        st.dataframe(
            pd.DataFrame(incident["topology"]["edges"], columns=["Source", "Target"]),
            use_container_width=True,
            hide_index=True,
        )
    with col2:
        st.subheader("Alarms")
        st.dataframe(pd.DataFrame({"Alarm": incident["alarms"]}), use_container_width=True, hide_index=True)
        st.subheader("Latest KPIs")
        st.dataframe(_kpi_table(incident["kpis"]), use_container_width=True, hide_index=True)

with tab_trace:
    trace_rows = []
    for idx, step in enumerate(report["trace"], start=1):
        trace_rows.append(
            {
                "Step": idx,
                "Agent": step["agent"],
                "Action": step["action"],
                "Summary": step["summary"],
                "Tool calls": len(step["tool_calls"]),
            }
        )
    st.dataframe(pd.DataFrame(trace_rows), use_container_width=True, hide_index=True)
    selected_step = st.selectbox("Trace detail", [f"{row['Step']}. {row['Agent']}" for row in trace_rows])
    step_index = int(selected_step.split(".", 1)[0]) - 1
    st.json(report["trace"][step_index])

with tab_evidence:
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("Selected Root Cause")
        st.write(report["root_cause"])
        st.subheader("Evidence")
        for item in report["evidence"]:
            st.markdown(f"- {item}")
    with col2:
        st.subheader("Hypothesis Scores")
        score_rows = []
        for hypothesis in report["hypotheses"]:
            row = {"Cause": hypothesis["cause"], **hypothesis["scores"]}
            score_rows.append(row)
        st.dataframe(pd.DataFrame(score_rows), use_container_width=True, hide_index=True)
        st.subheader("Ground Truth")
        st.write(incident["ground_truth"])

with tab_remediation:
    col1, col2 = st.columns([1.1, 1])
    with col1:
        st.subheader("Recommended Actions")
        for index, action in enumerate(report["recommended_actions"], start=1):
            st.markdown(f"{index}. {action}")
        st.subheader("Validation Plan")
        for rule in report["validation_plan"]:
            st.markdown(f"- {rule}")
    with col2:
        st.subheader("Validation Result")
        result = report["validation_result"]
        st.metric("Status", result["status"])
        st.dataframe(
            pd.DataFrame(result["validation_result"].get("checks", [])),
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download JSON report",
            data=json.dumps(report, indent=2),
            file_name=f"{incident_id}_report.json",
            mime="application/json",
        )
