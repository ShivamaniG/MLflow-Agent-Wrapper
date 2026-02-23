import agent_router
from mlflow.genai.agent_server import AgentServer

agent_server = AgentServer("ResponsesAgent")
app = agent_server.app

def main() -> None:
    agent_server.run(app_import_string="start_server:app")


if __name__ == "__main__":
    main()
