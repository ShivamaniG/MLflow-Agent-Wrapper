import os
from typing import TypedDict

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from mlflow.genai.agent_server import invoke
from mlflow.types.responses import ResponsesAgentRequest, ResponsesAgentResponse


load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "Test"))
mlflow.langchain.autolog()


class AgentState(TypedDict):
    question: str
    careplan: str


def _build_llm() -> ChatGoogleGenerativeAI:
    gateway_base = os.getenv("AI_GATEWAY_BASE_URL", "http://localhost:5000/gateway/gemini")
    gateway_endpoint = os.getenv("AI_GATEWAY_GEMINI_ENDPOINT", "Test_Endpoint")
    gateway_api_key = os.getenv("AI_GATEWAY_API_KEY", "dummy")

    return ChatGoogleGenerativeAI(
        model=gateway_endpoint,
        google_api_key=gateway_api_key,
        base_url=gateway_base,
        temperature=0.1,
    )


def _careplan_node(state: AgentState) -> AgentState:
    llm = _build_llm()

    system = SystemMessage(
        content=(
            "You are a medical assistant. Return exactly one short line care plan. "
            "No bullets. No disclaimer."
        )
    )
    user = HumanMessage(content=state["question"])
    response = llm.invoke([system, user])

    state["careplan"] = str(response.content).strip().replace("\n", " ")
    return state


workflow = StateGraph(AgentState)
workflow.add_node("careplan", _careplan_node)
workflow.set_entry_point("careplan")
workflow.add_edge("careplan", END)
medical_graph = workflow.compile()


def run_medical_agent(question: str) -> str:
    result = medical_graph.invoke({"question": question, "careplan": ""})
    return result["careplan"]


@invoke()
async def non_streaming(request: ResponsesAgentRequest) -> ResponsesAgentResponse:
    question = ""
    for item in request.input:
        payload = item.model_dump() if hasattr(item, "model_dump") else item
        if isinstance(payload, dict):
            content = payload.get("content")
            if isinstance(content, str) and content.strip():
                question = content.strip()
                break

    if not question:
        return ResponsesAgentResponse(custom_outputs={"careplan": "Missing user question."}, output=[])

    careplan = run_medical_agent(question)
    return ResponsesAgentResponse(custom_outputs={"careplan": careplan}, output=[])


if __name__ == "__main__":
    q = "Adult with fever, cough, and mild dehydration for 2 days."
    print(run_medical_agent(q))
