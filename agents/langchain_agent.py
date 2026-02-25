import os
from typing import TypedDict

import mlflow
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph


load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("langchain")
mlflow.langchain.autolog()


class AgentState(TypedDict):
    question: str
    careplan: str


def _build_llm() -> ChatGoogleGenerativeAI:
    gateway_base = os.getenv("AI_GATEWAY_BASE_URL", "http://localhost:5000/gateway/gemini")
    gateway_endpoint = os.getenv("AI_GATEWAY_GEMINI_ENDPOINT", "Gemini-Endpoint")
    gateway_api_key = os.getenv("AI_GATEWAY_API_KEY", "dummy")

    return ChatGoogleGenerativeAI(
        model=gateway_endpoint,
        google_api_key=gateway_api_key,
        base_url=gateway_base,
        temperature=0.1,
    )


def _careplan_node(state: AgentState) -> AgentState:
    llm = _build_llm()
    prompt = mlflow.genai.load_prompt("prompts:/v1/1")
    prompt_text = getattr(prompt, "template", str(prompt))

    response = llm.invoke(
        [
            SystemMessage(content=prompt_text),
            HumanMessage(content=state["question"]),
        ]
    )

    state["careplan"] = str(response.content).strip().replace("\n", " ")
    return state



workflow = StateGraph(AgentState)
workflow.add_node("careplan", _careplan_node)
workflow.set_entry_point("careplan")
workflow.add_edge("careplan", END)
medical_graph = workflow.compile()


def run_langchain_agent(question: str) -> str:
    result = medical_graph.invoke({"question": question, "careplan": ""})
    return result["careplan"]


if __name__ == "__main__":
    q = "Adult with fever, cough, and mild dehydration for 2 days."
    print(run_langchain_agent(q))
