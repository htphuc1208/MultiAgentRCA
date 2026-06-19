# Multi-Agent AI Telecom RCA

This repository implements a telecom root-cause analysis prototype with two modes:

- `llm`: AI-backed multi-agent RCA using OpenAI structured outputs and tool calling.
- `rule`: deterministic offline fallback for tests and demos without an API key.

The main workflow is:

```text
Incident -> Orchestrator -> Triage -> Data Retrieval -> Topology
         -> RCA Hypotheses -> SOP Retrieval -> Consensus/Verifier
         -> Remediation Planner -> Validation -> RCA Report
```

## Setup

Install dependencies:

```bash
python3 -m pip install --user --break-system-packages -r requirements.txt
```

For AI mode:

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-5.5"
export OPENAI_REASONING_EFFORT="medium"
```

## Run

List incidents:

```bash
python3 -m app.cli --list
```

Run the real AI-agent workflow:

```bash
python3 -m app.cli --incident-id INC-RAN-001 --output reports/INC-RAN-001.json
```

Run offline fallback:

```bash
python3 -m app.cli --mode rule --incident-id INC-RAN-001
```

Start the UI:

```bash
streamlit run ui/streamlit_app.py
```

## Evaluation

Offline smoke evaluation:

```bash
python3 -m app.evaluation.run_experiments --mode rule
```

LLM evaluation with repeated runs for stability:

```bash
python3 -m app.evaluation.run_experiments --mode llm --repeats 3
```

The runner writes:

- `reports/experiment_results.csv`
- `reports/experiment_summary.csv`

Configurations:

- `Baseline 1 Rule/SOP lookup only`
- `Baseline 2 Single ReAct-style agent`
- `Baseline 3 Multi-Agent without consensus`
- `Proposed Multi-Agent + SOP + Consensus`

## Data Model

Runtime incident data lives in `data/incidents.json`. Hidden evaluation labels are separated into `data/eval_labels.json`, and ticket history is separated into `data/tickets.json`.

`DataStore.get_incident()` strips these fields from runtime input:

- `ground_truth`
- `expected_actions`
- `sop_id`
- `ticket_history`

This prevents LLM agents from reading labels or SOP shortcuts during RCA.

## Report

The RCA report includes:

- selected root cause and confidence
- evidence and evidence refs
- ranked hypotheses
- selected SOP
- verification notes
- recommended remediation actions
- validation plan and validation result
- full agent/tool trace
- LLM calls, token usage, and latency

## Notes

`llm` mode is the intended AI-agent system. `rule` mode is retained only for offline reproducibility and regression tests.

