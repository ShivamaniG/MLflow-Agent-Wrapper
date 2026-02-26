
# MLflow Agent Observability Router

This service is a modular MLflow-powered agent router designed to help teams **observe, monitor, and evaluate agent performance in a structured way**.

Instead of running isolated agents without visibility, this framework provides:

* Centralized agent routing
* Prompt and response tracking
* Experiment-level observability
* Per-agent monitoring via MLflow

It enables you to understand how agents behave in real time, how prompts evolve, and how outputs perform across experiments.

---

## The Problem We’re Solving

When building LLM agents, teams often face:

* No visibility into prompt behavior
* No experiment tracking per agent
* No structured observability
* Difficulty debugging production responses
* No standardized evaluation pipeline

This service solves that by placing MLflow at the center of agent execution.

---


## How It Works

All agents are routed through a single `ResponsesAgent` service (or can be invoked via your own API service). Regardless of how the agent is triggered, observability is centrally tracked in the MLflow server.

Each agent:

* Is registered declaratively
* Runs independently
* Logs to its own MLflow experiment
* Supports autologging for prompts, inputs, outputs, latency, and metadata

This creates a unified observability layer where every agent execution is traceable, measurable, and comparable inside MLflow.

---

## AI Gateway Integration

Model access is abstracted through the MLflow AI Gateway. This allows you to change the underlying LLM (e.g., Gemini, OpenAI, etc.) directly from the Gateway UI without modifying agent code.

Agents remain unchanged while the model configuration can be updated centrally. This enables:

* Model switching without code changes
* Centralized provider management
* Secure API key handling
* Consistent routing through a standard endpoint

The agent logic stays stable, while model experimentation and upgrades can be managed independently through the Gateway.

---

## Observability with MLflow

By configuring MLflow in each agent:

```python
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("EXPERIMENT_NAME")
mlflow.agno.autolog()
```

You automatically gain:

* Prompt tracking
* Input/output logging
* Token usage visibility
* Latency metrics
* Experiment comparisons
* Per-agent performance monitoring

This makes agent testing measurable and production-ready.

---

## Agent registry details

- `agents/agents.json` now only holds `agent_id` + `experiment`; the loader derives the module name `agents.{agent_id}_agent` and runner `run_{agent_id}_agent` automatically.
- Name your implementation files and runner functions accordingly (e.g., `summary_agent.py` defines `run_summary_agent`) so the registry can import them without extra metadata.
- The router shares the same `ResponsesAgent` contract shown above; clients select agents by sending their `agent_id` in `custom_inputs`.

## Agent Testing (API Contract)

To test any agent:

**POST** `http://localhost:8000/agent/responses`

```json
{
  "model": "ResponsesAgent",
  "input": [],
  "custom_inputs": {
    "agent_id": "langchain",
    "payload": {
      "content": "How can I improve my heart health?"
    }
  }
}
```

### Sample Response

```json
{
  "object": "response",
  "output": [],
  "custom_outputs": {
    "agent_id": "langchain",
    "agent_output": "Incorporate daily moderate exercise and a diet rich in fruits, vegetables, and whole grains.",
    "status": "success"
  }
}
```
