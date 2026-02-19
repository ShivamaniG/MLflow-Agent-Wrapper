# MLflow Medical Agent Wrapper - Version 1

## What this is
A minimal MLflow Agent Server wrapper around a LangGraph medical agent.

- Input: user medical question
- Output: one-line care plan
- Model path: Gemini via MLflow AI Gateway

## Why use this
- Exposes your agent as an HTTP API (`/invocations`)
- Keeps model/provider details behind MLflow Gateway
- Adds MLflow tracing compatibility for observability

## Project files
- `agent.py`: LangGraph agent + `@invoke()` handler
- `start_server.py`: starts MLflow `AgentServer`
- `.env`: runtime configuration
- `requirements.txt`: dependencies

## Prerequisites
- Python 3.10+
- MLflow server running at `http://localhost:5000`
- A Gemini endpoint configured in MLflow AI Gateway

## Environment config
Set `.env`:

```env
AI_GATEWAY_BASE_URL=http://localhost:5000/gateway/gemini
AI_GATEWAY_GEMINI_ENDPOINT=Test_Endpoint
AI_GATEWAY_API_KEY=dummy
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT=Test
```

Notes:
- `AI_GATEWAY_GEMINI_ENDPOINT` must exactly match your MLflow Gateway endpoint name.
- Keep `AI_GATEWAY_API_KEY=dummy` only if your gateway does not enforce auth.

## Install
```bash
pip install -r requirements.txt
```

## Start
1. Start MLflow server:
```bash
mlflow server --host 127.0.0.1 --port 5000
```

2. Start agent server:
```bash
python start_server.py
```

## Test API
```bash
curl -X POST http://localhost:8000/invocations \
  -H "Content-Type: application/json" \
  -d '{"input":[{"role":"user","content":"Adult with mild fever and sore throat for 2 days"}]}'
```

Expected shape:
```json
{
  "custom_outputs": {
    "careplan": "...one line plan..."
  }
}
```

## Version
Version 1: minimal single-agent wrapper (no multi-agent routing, no RAG).
