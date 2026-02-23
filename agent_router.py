from dataclasses import dataclass
from typing import Any, Callable, Dict

from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agno_agent import run_agno_agent
from langchain_agent import run_langchain_agent


@dataclass(frozen=True)
class AgentRequestData:
    agent_id: str
    payload: Dict[str, Any]


def _extract_content(agent_input: Dict[str, Any]) -> str:
    content = agent_input.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("payload.content must be a non-empty string.")
    return content.strip()


def _build_agent_registry() -> Dict[str, Callable[[Dict[str, Any]], str]]:
    return {
        "agno": lambda payload: run_agno_agent(_extract_content(payload)),
        "langchain": lambda payload: run_langchain_agent(_extract_content(payload)),
    }


AGENT_REGISTRY = _build_agent_registry()


def _parse_conv_request(request: ResponsesAgentRequest) -> AgentRequestData:
    custom_inputs = getattr(request, "custom_inputs", None)
    if hasattr(custom_inputs, "model_dump"):
        custom_inputs = custom_inputs.model_dump()
    if not isinstance(custom_inputs, dict):
        raise ValueError("custom_inputs must be a dict with agent_id and payload fields.")

    agent_id = custom_inputs.get("agent_id")
    payload = custom_inputs.get("payload")
    if not isinstance(agent_id, str) or not agent_id.strip():
        raise ValueError("agent_id must be a non-empty string.")
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict describing the agent request.")

    return AgentRequestData(agent_id=agent_id.strip().lower(), payload=payload)


def _build_response(
    agent_id: str, agent_output: str, status: str = "success", message: str = ""
) -> ResponsesAgentResponse:
    payload = {"agent_id": agent_id, "agent_output": agent_output, "status": status}
    if message:
        payload["message"] = message
    return ResponsesAgentResponse(custom_outputs=payload, output=[])


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    try:
        payload = _parse_conv_request(request)
        agent_fn = AGENT_REGISTRY.get(payload.agent_id)
        if agent_fn is None:
            raise ValueError(f"Unknown agent_id '{payload.agent_id}'.")

        agent_response = agent_fn(payload.payload)
        return _build_response(payload.agent_id, agent_response)
    except Exception as exc:
        return _build_response("unknown", "", status="error", message=str(exc))
