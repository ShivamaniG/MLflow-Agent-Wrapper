# MLflow Medical Agent Router

## What this is
An MLflow `ResponsesAgent` wrapper that routes incoming prompts to modular medical agents. Each agent is defined in the `agents/` package and registered via `agents.registry`. The router exposes a single `/agent/responses` endpoint backed by `mlflow.genai.AgentServer`.

## Why use it
- Unified API surface for multiple medical agents (Agno, LangGraph, etc.).
- Agents are autologged through MLflow experiments (one per agent).
- The router validates payloads, normalizes the `agent_id`, and replies with a consistent `custom_outputs` payload.
- New agents plug in by adding a config entry—no handler rewrites.

- `agents/`: package containing agent implementations and the registry config.
- `agents/agents.json`: declarative agent metadata (`agent_id`, module path, runner, experiment name) that `agents.registry` loads at startup.
- `agents.registry`: loader that builds `AGENT_REGISTRY` from `agents.json`.
- `agent_router.py`: `@invoke`-decorated handler that dispatches requests to the registered agent callable.
- `start_server.py`: boots `mlflow.genai.AgentServer("ResponsesAgent")` on port 8000.
- `.env(.example)`: configuration for the MLflow AI Gateway & tracking server.
- `requirements.txt`: runtime dependencies (including `mlflow`, `agno`, `langchain-google-genai`, etc.).

## Environment
Create `.env` (copy from `.env.example`) with your gateway settings:

```env
AI_GATEWAY_BASE_URL=http://localhost:5000/gateway/gemini
AI_GATEWAY_GEMINI_ENDPOINT=Gemini_Endpoint
AI_GATEWAY_API_KEY=dummy
MLFLOW_TRACKING_URI=http://localhost:5000
```

The `AI_GATEWAY_GEMINI_ENDPOINT` must match the name registered with your gateway, and `MLFLOW_TRACKING_URI` should point to your tracking server (default `http://localhost:5000`).

## Install
```bash
pip install -r requirements.txt
```

## Running
1. Launch MLflow server if you rely on autologging: `mlflow server --host 127.0.0.1 --port 5000`.
2. Start the agent router: `python start_server.py`.
   The server runs `AgentServer("ResponsesAgent")` on `http://0.0.0.0:8000`.

## API contract
`POST http://localhost:8000/agent/responses` with:

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

Sample response:

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

Errors set `status: "error"` in `custom_outputs` and include a `message`.

## Adding agents
1. Add the implementation module under `agents/` and expose a runner function (e.g., `run_my_agent(prompt: str) -> str`).
2. Add a JSON entry to `agents/agents.json` with your `agent_id`, module path, runner name, and `experiment` tag.
3. Restart `start_server.py`. The router reads `agents/agents.json`, loads each runner, and makes it available to clients via `custom_inputs.agent_id`.

## Version
Version 2: modular ResponsesAgent router with a declarative agent registry.
