import os
import json
from pathlib import Path

import cohere
from dotenv import load_dotenv


load_dotenv()

if not os.getenv("CO_API_KEY"):
    raise RuntimeError(
        "Missing CO_API_KEY. Create a .env file with CO_API_KEY=... (see .env.example)."
    )

co = cohere.Client(api_key=os.environ["CO_API_KEY"])

response = co.chat(
    model="command-a-03-2025",
    message="Tell me about LLMs",
)

def _jsonable(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "dict"):
        try:
            return value.dict()
        except TypeError:
            pass
    return value


out_dir = Path("outputs")
out_dir.mkdir(exist_ok=True)

out_path = out_dir / "helloworld_chat_response.json"
out_path.write_text(
    json.dumps(_jsonable(response), indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)

print(f"Wrote response to {out_path}")
