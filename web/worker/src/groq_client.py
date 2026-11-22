from __future__ import annotations

import json
from typing import Any

from js import Object, fetch
from pyodide.ffi import to_js

from chat import extract_answer


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqRequestError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Groq returned HTTP {status_code}.")
        self.status_code = status_code


def to_javascript(value: Any) -> Any:
    return to_js(value, dict_converter=Object.fromEntries)


async def request_completion(
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.5,
    }
    options = to_javascript({
        "method": "POST",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    })

    response = await fetch(GROQ_CHAT_COMPLETIONS_URL, options)
    if not response.ok:
        raise GroqRequestError(int(response.status))

    payload = json.loads(await response.text())
    return extract_answer(payload)
