# Multi-Agent Telecom RCA Prototype

This repository implements a decision-support prototype for telecom incident RCA:

- multi-agent workflow with orchestrator, triage, data retrieval, topology, RCA, SOP, verifier, planner, and validation agents
- mocked tool layer for alarms, KPIs, logs, topology, ticket history, diagnostics, SOP retrieval, and post-fix validation
- shared blackboard trace for every agent action and tool call
- consensus scoring over evidence, topology, SOP alignment, history, and agent confidence
- synthetic RAN, Core, and Transport/IP incident dataset with ground truth
- CLI, Streamlit demo, and experiment runner

## Run

List incidents:

```bash
python3 -m app.cli --list
```

Generate one RCA report:

```bash
python3 -m app.cli --incident-id INC-RAN-001 --output reports/INC-RAN-001.json
```

Run baseline/proposed experiments:

```bash
python3 -m app.evaluation.run_experiments
```

Start the demo UI:

```bash
streamlit run ui/streamlit_app.py
```

## Architecture

The workflow follows:

```text
Incident -> Orchestrator -> Triage -> Data Retrieval -> Topology
         -> RCA Hypotheses -> SOP Retrieval -> Consensus/Verifier
         -> Remediation Planner -> Validation -> Incident Report
```

The final report exposes root cause, confidence, supporting evidence, ranked hypotheses, remediation actions, validation checks, and a full agent/tool trace.

## Evaluation Metrics

`app/evaluation/metrics.py` calculates:

- RCA accuracy
- Top-3 RCA accuracy
- remediation correctness
- tool-use validity
- evidence coverage
- hallucination rate
- stability

The experiment runner writes:

- `reports/experiment_results.csv`
- `reports/experiment_summary.csv`

## Notes

The current implementation is deterministic and does not require an LLM API key. This keeps the prototype reproducible for demos and research evaluation. A real LLM planner can later replace the rule-based `RCAAgent` while preserving the same tool, blackboard, verifier, and evaluation interfaces.

