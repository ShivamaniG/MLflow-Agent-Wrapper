import os

import mlflow
from agno.agent import Agent
from agno.models.google import Gemini
from dotenv import load_dotenv

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("agno")
mlflow.agno.autolog()

def _load_instructions() -> str:
    prompt = mlflow.genai.load_prompt("prompts:/v1/1")
    return getattr(prompt, "template", str(prompt))


def _build_agent(instructions: str) -> Agent:
    model = Gemini(
        id=os.getenv("AI_GATEWAY_GEMINI_ENDPOINT", "Gemini-Endpoint"),
        api_key=os.getenv("AI_GATEWAY_API_KEY", "dummy"),
        client_params={
            "http_options": {
                "base_url": os.getenv(
                    "AI_GATEWAY_BASE_URL", "http://localhost:5000/gateway/gemini"
                )
            }
        },
    )

    return Agent(
        name="Medical Agent",
        model=model,
        instructions=instructions,
        markdown=False,
    )


INSTRUCTIONS = _load_instructions()
AGENT = _build_agent(INSTRUCTIONS)


def run_agno_agent(question: str) -> str:
    q = question.strip()
    response = AGENT.run(input=q)
    answer = str(getattr(response, "content", response)).strip()
    return answer


def main() -> None:
    question = "I want to improve heart health"
    print(run_agno_agent(question))


if __name__ == "__main__":
    main()
