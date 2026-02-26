import os
import base64
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import mlflow

load_dotenv()

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("summary-agent")
mlflow.langchain.autolog()


class MedicalDocumentAgent:
    def __init__(self):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env")

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key
        )

    def _build_prompt(self):
        return (
            "You are a medical document reviewer.\n\n"
            "Read the attached PDF and provide:\n"
            "1. A clear and concise medical summary.\n"
            "2. A bullet list of key diagnoses, medications, procedures, and lab results mentioned.\n\n"
            "Important:\n"
            "- Use only information found in the document.\n"
            "- Do not add or assume anything.\n"
            "- Do not mention missing information.\n"
            "- Keep the response professional and easy to understand.\n"
        )

    def _pdf_to_base64(self, pdf_path: str) -> str:
        with open(pdf_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def process_document(self, pdf_path: str) -> dict:
        try:
            pdf_base64 = self._pdf_to_base64(pdf_path)
            prompt = self._build_prompt()

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "file",
                        "source_type": "base64",
                        "mime_type": "application/pdf",
                        "data": pdf_base64,
                        "filename": os.path.basename(pdf_path)
                    }
                ]
            )

            response = self.llm.invoke([message])

            return {
                "status": "success",
                "model": "gemini-2.5-flash",
                "output": response.content
            }

        except Exception as e:
            return {
                "status": "error",
                "message": str(e)
            }


if __name__ == "__main__":
    agent = MedicalDocumentAgent()
    result = agent.process_document("../data/AUTH Request for ADHD Pharma.pdf")
    print(json.dumps(result, indent=2))


def run_medical_summary_agent(pdf_path: str) -> str:
    agent = MedicalDocumentAgent()
    result = agent.process_document(pdf_path)
    if result.get("status") != "success":
        raise RuntimeError(result.get("message", "Medical summary agent failed"))
    return result["output"]
