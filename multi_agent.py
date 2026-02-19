from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse

from agno_agent import run_agno_agent
from langchain_agent import run_medical_agent


def _extract_question(request: ResponsesAgentRequest) -> str:
    for item in request.input:
        payload = item.model_dump() if hasattr(item, "model_dump") else item
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def _extract_agent_name(request: ResponsesAgentRequest) -> str:
    custom_inputs = getattr(request, "custom_inputs", None)
    if hasattr(custom_inputs, "model_dump"):
        custom_inputs = custom_inputs.model_dump()
    if isinstance(custom_inputs, dict):
        value = custom_inputs.get("agent_name")
        if isinstance(value, str):
            return value.strip().lower()

    context = getattr(request, "context", None)
    if hasattr(context, "model_dump"):
        context = context.model_dump()
    if isinstance(context, dict):
        value = context.get("agent_name")
        if isinstance(value, str):
            return value.strip().lower()

    metadata = getattr(request, "metadata", None)
    if hasattr(metadata, "model_dump"):
        metadata = metadata.model_dump()
    if isinstance(metadata, dict):
        value = metadata.get("agent_name")
        if isinstance(value, str):
            return value.strip().lower()
    return "agno"


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    question = _extract_question(request) or "I want to improve heart health"
    agent_name = _extract_agent_name(request)

    if agent_name in {"langchain", "langgraph"}:
        answer = run_medical_agent(question)
        selected = "langchain"
    else:
        answer = run_agno_agent(question)
        selected = "agno"

    return ResponsesAgentResponse(
        custom_outputs={"answer": answer, "agent_name": selected},
        output=[],
    )
