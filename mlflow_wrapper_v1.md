# MLflow Medical Agent Wrapper - Version 1

## What this is
A minimal MLflow Agent Server wrapper with **two Gemini agents** behind one `/invocations` endpoint:
- `agno` agent (`agno_agent.py`)
- `langchain` (LangGraph) agent (`langchain_agent.py`)

Both call Gemini via MLflow AI Gateway and return one concise care-plan line.

## Why use this
- Single API endpoint for multiple agent frameworks
- Gateway-based model access (no direct provider SDK calls in request layer)
- MLflow autologging enabled for each framework:
  - `mlflow.agno.autolog()` in Agno agent
  - `mlflow.langchain.autolog()` in LangGraph agent
- Separate experiments:
  - Agno agent logs to experiment `agno`
  - LangGraph agent logs to experiment `langchain`

## Project files
- `agno_agent.py`: Agno medical agent logic
- `langchain_agent.py`: LangGraph medical agent logic
- `multi_agent.py`: single `@invoke()` router for both agents
- `start_server.py`: starts MLflow `AgentServer`
- `.env`: runtime configuration
- `.env.example`: sample configuration
- `requirements.txt`: dependencies

## Environment config
Set `.env` (or copy from `.env.example`):

```env
AI_GATEWAY_BASE_URL=http://localhost:5000/gateway/gemini
AI_GATEWAY_GEMINI_ENDPOINT=Gemini_Endpoint
AI_GATEWAY_API_KEY=dummy
MLFLOW_TRACKING_URI=http://localhost:5000
```

Notes:
- `AI_GATEWAY_GEMINI_ENDPOINT` must match your MLflow Gateway endpoint name exactly.
- `AI_GATEWAY_API_KEY=dummy` is fine if your local gateway does not enforce auth.

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

## API usage
Use `custom_inputs.agent_name` to choose agent.

### Call Agno agent
```bash
curl -X POST http://localhost:8000/invocations   -H "Content-Type: application/json"   -d '{
    "input": [{"role":"user","content":"I want to improve heart health"}],
    "custom_inputs": {"agent_name":"agno"}
  }'
```

### Call LangGraph agent
```bash
curl -X POST http://localhost:8000/invocations   -H "Content-Type: application/json"   -d '{
    "input": [{"role":"user","content":"I want to improve heart health"}],
    "custom_inputs": {"agent_name":"langchain"}
  }'
```

## Response shape
```json
{
  "object": "response",
  "output": [],
  "custom_outputs": {
    "answer": "...one line plan...",
    "agent_name": "agno or langchain"
  }
}
```

## Version
Version 1: minimal multi-agent wrapper (Agno + LangGraph), Gemini only, no RAG.
