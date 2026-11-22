from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import urlparse

from workers import Response, WorkerEntrypoint

from chat import InvalidConversation, build_messages
from events import log_event
from groq_client import GroqRequestError, request_completion
from usage import reserve_monthly_call


DEFAULT_MONTHLY_CALL_LIMIT = 900
MAX_MONTHLY_CALL_LIMIT = 900
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"


def read_monthly_limit(configured_value: object) -> int:
    try:
        configured_limit = int(configured_value)
    except (TypeError, ValueError):
        return DEFAULT_MONTHLY_CALL_LIMIT

    if not 1 <= configured_limit <= MAX_MONTHLY_CALL_LIMIT:
        return DEFAULT_MONTHLY_CALL_LIMIT

    return configured_limit


def json_response(payload: dict[str, object], status: int = 200) -> Response:
    return Response.json(payload, status=status)


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        parsed_url = urlparse(request.url)

        if parsed_url.path == "/api/health":
            if request.method != "GET":
                return json_response({"error": "Method not allowed."}, 405)

            return json_response({
                "status": "ok",
                "chat_configured": bool(getattr(self.env, "GROQ_API_KEY", "")),
                "rag_enabled": False,
            })

        if parsed_url.path == "/api/chat":
            if request.method != "POST":
                return json_response({"error": "Method not allowed."}, 405)

            return await self._handle_chat(request)

        if parsed_url.path.startswith("/api/"):
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
            return json_response(
                {"error": "The chat service is not configured yet."},
                503,
            )

        now_utc = datetime.now(timezone.utc)
        month_utc = now_utc.strftime("%Y-%m")
        monthly_limit = read_monthly_limit(
            getattr(self.env, "MONTHLY_CALL_LIMIT", DEFAULT_MONTHLY_CALL_LIMIT)
        )

        try:
            call_reserved = await reserve_monthly_call(
                self.env.USAGE_DB,
                month_utc,
                monthly_limit,
            )
        except Exception as error:
            log_event(
                "USAGE_COUNTER_FAILED",
                error_type=type(error).__name__,
            )
            return json_response(
                {"error": "Usage tracking is temporarily unavailable."},
                503,
            )

        if not call_reserved:
            log_event("MONTHLY_CALL_LIMIT_REACHED", month=month_utc)
            return json_response(
                {"error": "This demo has reached its monthly message limit."},
                429,
            )

        model = getattr(self.env, "GROQ_MODEL", DEFAULT_GROQ_MODEL)

        try:
            answer = await request_completion(
                api_key=api_key,
                model=model,
                messages=messages,
            )
        except GroqRequestError as error:
            log_event(
                "GROQ_REQUEST_FAILED",
                status=error.status_code,
                model=model,
            )
            return json_response(
                {"error": "The AI provider request failed. Check the server logs."},
                502,
            )
        except Exception as error:
            log_event(
                "GROQ_RESPONSE_INVALID",
                error_type=type(error).__name__,
                model=model,
            )
            return json_response(
                {"error": "The AI provider returned an invalid response."},
                502,
            )

        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        log_event(
            "CHAT_COMPLETED",
            model=model,
            history_messages=len(messages) - 1,
            duration_ms=elapsed_ms,
        )
        return json_response({"answer": answer})
