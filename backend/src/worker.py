from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from js import Object, fetch
from pyodide.ffi import to_js
from workers import Response, WorkerEntrypoint

from chatbot.chat_service import (
    InvalidConversation,
    build_messages,
    extract_answer,
)
from db.usage_counter import reserve_monthly_call
from log.worker_events import log_event


GROQ_CHAT_COMPLETIONS_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MONTHLY_CALL_LIMIT = 900
MAX_MONTHLY_CALL_LIMIT = 900
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


class GroqRequestError(Exception):
    def __init__(self, status_code: int) -> None:
        super().__init__(f"Groq returned HTTP {status_code}.")
        self.status_code = status_code


def read_monthly_limit(configured_value: object) -> int:
    try:
        configured_limit = int(configured_value)
    except (TypeError, ValueError):
        return DEFAULT_MONTHLY_CALL_LIMIT
    return (
        configured_limit
        if 1 <= configured_limit <= MAX_MONTHLY_CALL_LIMIT
        else DEFAULT_MONTHLY_CALL_LIMIT
    )


def json_response(payload: dict[str, object], status: int = 200) -> Response:
    return Response.json(payload, status=status)


async def request_completion(api_key: str, model: str, messages: list[dict[str, str]]) -> str:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.5,
    }
    options = to_js({
        "method": "POST",
        "headers": {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        "body": json.dumps(body),
    }, dict_converter=Object.fromEntries)

    response = await fetch(GROQ_CHAT_COMPLETIONS_URL, options)
    if not response.ok:
        raise GroqRequestError(int(response.status))
    return extract_answer(json.loads(await response.text()))


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        path = urlparse(request.url).path
        if path == "/api/health":
            if request.method != "GET":
                return json_response({"error": "Method not allowed."}, 405)
            return json_response({
                "status": "ok",
                "chat_configured": bool(getattr(self.env, "GROQ_API_KEY", "")),
                "rag_enabled": False,
            })

        if path == "/api/chat":
            if request.method != "POST":
                return json_response({"error": "Method not allowed."}, 405)
            return await self._handle_chat(request)

        if path.startswith("/api/"):
            return json_response({"error": "Not found."}, 404)
        return await self.env.ASSETS.fetch(request)

    async def _handle_chat(self, request):
        started_at = time.perf_counter()
        try:
            request_body = await request.json()
        except Exception:
            log_event("CHAT_REJECTED", reason="invalid_json")
            return json_response({"error": "Request body must be valid JSON."}, 400)

        try:
            messages = build_messages(request_body.get("messages"))
        except (AttributeError, InvalidConversation) as error:
            log_event("CHAT_REJECTED", reason="invalid_conversation")
            return json_response({"error": str(error)}, 400)

        api_key = getattr(self.env, "GROQ_API_KEY", "")
        if not api_key:
            log_event("CHAT_UNAVAILABLE", reason="api_key_not_configured")
            return json_response({"error": "The chat service is not configured yet."}, 503)

        month_utc = datetime.now(timezone.utc).strftime("%Y-%m")
        monthly_limit = read_monthly_limit(
            getattr(self.env, "MONTHLY_CALL_LIMIT", DEFAULT_MONTHLY_CALL_LIMIT)
        )
        try:
            reserved = await reserve_monthly_call(
                self.env.USAGE_DB, month_utc, monthly_limit
            )
        except Exception as error:
            log_event("USAGE_COUNTER_FAILED", error_type=type(error).__name__)
            return json_response({"error": "Usage tracking is temporarily unavailable."}, 503)

        if not reserved:
            log_event("MONTHLY_CALL_LIMIT_REACHED", month=month_utc)
            return json_response({"error": "This demo has reached its monthly message limit."}, 429)

        model = getattr(self.env, "GROQ_MODEL", DEFAULT_GROQ_MODEL)
        try:
            answer = await request_completion(api_key, model, messages)
        except GroqRequestError as error:
            log_event("GROQ_REQUEST_FAILED", status=error.status_code, model=model)
            return json_response(
                {"error": "The AI provider request failed. Check the server logs."}, 502
            )
        except Exception as error:
            log_event(
                "GROQ_RESPONSE_INVALID",
                error_type=type(error).__name__,
                model=model,
            )
            return json_response({"error": "The AI provider returned an invalid response."}, 502)

        log_event(
            "CHAT_COMPLETED",
            model=model,
            history_messages=len(messages) - 1,
            duration_ms=round((time.perf_counter() - started_at) * 1000, 2),
        )
        return json_response({"answer": answer})
