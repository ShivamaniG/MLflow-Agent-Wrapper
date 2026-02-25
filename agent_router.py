from dataclasses import dataclass
from typing import Any, Dict

from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agents.registry import AGENT_CONFIGS, AGENT_REGISTRY


@dataclass(frozen=True)
class AgentRequestData:
    agent_id: str
    payload: Dict[str, Any]


def _extract_content(payload: Dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("payload.content must be a non-empty string.")
    return content.strip()


def _parse_conv_request(request: ResponsesAgentRequest) -> AgentRequestData:
    custom_inputs = getattr(request, "custom_inputs", None)
    if hasattr(custom_inputs, "model_dump"):
        custom_inputs = custom_inputs.model_dump()
    if not isinstance(custom_inputs, dict):
        raise ValueError("custom_inputs must contain agent_id and payload fields.")

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
            available = ", ".join(agent.agent_id for agent in AGENT_CONFIGS)
            raise ValueError(f"Unknown agent_id '{payload.agent_id}'. Available: {available}.")

        agent_response = agent_fn(_extract_content(payload.payload))
        return _build_response(payload.agent_id, agent_response)
    except Exception as exc:
        return _build_response("unknown", "", status="error", message=str(exc))
