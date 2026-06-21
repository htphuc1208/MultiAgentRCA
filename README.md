# Multi-Agent AI Telecom RCA

This repository implements a telecom root-cause analysis prototype with two modes:

- `llm`: AI-backed multi-agent RCA using DeepSeek or OpenAI.
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

For AI mode with DeepSeek, which is the default provider:

```bash
export LLM_PROVIDER="deepseek"
export DEEPSEEK_API_KEY="..."
export DEEPSEEK_MODEL="deepseek-v4-flash"
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
export DEEPSEEK_THINKING="disabled"
```

DeepSeek is called through its OpenAI-compatible Chat Completions API. The default model is
`deepseek-v4-flash` with thinking disabled to keep the multi-agent workflow lower latency and
lower cost. `deepseek-chat` and `deepseek-reasoner` are legacy model names in the DeepSeek docs
scheduled for deprecation on 2026-07-24 15:59 UTC, and are not used as defaults.

OpenAI remains available as a fallback provider:

```bash
export LLM_PROVIDER="openai"
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
python3 -m app.cli --mode llm --provider deepseek --incident-id INC-RAN-001 --output reports/INC-RAN-001.json
```

Run with OpenAI instead:

```bash
python3 -m app.cli --mode llm --provider openai --incident-id INC-RAN-001
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
python3 -m app.evaluation.run_experiments --mode llm --provider deepseek --repeats 3
```

The runner writes:

- `reports/experiment_results.csv`
- `reports/experiment_summary.csv`

Configurations:

- `Baseline 1 Rule/SOP lookup only`
- `Baseline 2 Single ReAct-style agent`
- `Baseline 3 Multi-Agent without consensus`
- `Proposed Multi-Agent + SOP + Consensus`

## OpenRCA Telecom Benchmark

This repo also includes a separate OpenRCA Telecom benchmark mode. It does not import OpenRCA rows into
`data/incidents.json`; it reads the official OpenRCA `query.csv` and telemetry files through bounded tools.

Download the OpenRCA dataset from the official repository instructions:

- https://github.com/microsoft/OpenRCA

Keep the large dataset outside git, then point `OPENRCA_DATA_DIR` at the directory that contains
`Telecom/query.csv` and `Telecom/telemetry/`:

```bash
export OPENRCA_DATA_DIR="/path/to/OpenRCA/dataset"
```

Run one Telecom query row:

```bash
python3 -m app.openrca.cli --dataset Telecom --row-id 0 --mode llm --provider deepseek
```

Run a small batch:

```bash
python3 -m app.openrca.cli --dataset Telecom --start-row 0 --limit 10 --output-dir reports/openrca
```

The runner writes:

- `reports/openrca/telecom_predictions.csv`
- `reports/openrca/telecom_evaluation.csv`
- `reports/openrca/telecom_summary.csv`

Runtime LLM calls only receive the query instruction, candidate catalog, telemetry dates, and tool-returned
metric/trace/log summaries. `scoring_points` stays out of runtime prompts and is used only after predictions
are generated for OpenRCA-compatible evaluation.

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
