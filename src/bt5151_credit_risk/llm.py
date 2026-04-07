import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from bt5151_credit_risk.config import DEFAULT_OPENAI_MODEL


load_dotenv()


def call_json_response(system_prompt: str, payload: dict) -> dict:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to .env before running the graph.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    response = client.responses.create(
        model=model,
        instructions=system_prompt,
        input=json.dumps(payload, ensure_ascii=True, indent=2),
    )
    if not response.output_text:
        raise RuntimeError("OpenAI response did not include output_text.")
    return json.loads(response.output_text)
