from __future__ import annotations

from typing import Any

from .config import SYSTEM_PROMPT


MAX_HISTORY_MESSAGES = 12
MAX_MESSAGE_CHARS = 4_000
MAX_HISTORY_CHARS = 16_000


class InvalidConversation(ValueError):
    """The browser supplied a malformed or oversized chat history."""


def build_messages(client_messages: Any) -> list[dict[str, str]]:
    if not isinstance(client_messages, list) or not client_messages:
        raise InvalidConversation("Send at least one user message.")

    if len(client_messages) > MAX_HISTORY_MESSAGES:
        raise InvalidConversation("Conversation is too long. Start a new chat.")

    messages: list[dict[str, str]] = []
    total_chars = 0

    for message in client_messages:
        if not isinstance(message, dict):
            raise InvalidConversation("Each message must be an object.")

        role = message.get("role")
        content = message.get("content")
        if role not in {"user", "assistant"}:
            raise InvalidConversation("Unsupported message role.")
        if not isinstance(content, str) or not content.strip():
            raise InvalidConversation("Messages cannot be empty.")

        content = content.strip()
        if len(content) > MAX_MESSAGE_CHARS:
            raise InvalidConversation("A message is too long.")

        total_chars += len(content)
        if total_chars > MAX_HISTORY_CHARS:
            raise InvalidConversation("Conversation is too long. Start a new chat.")

        messages.append({"role": role, "content": content})

    if messages[-1]["role"] != "user":
        raise InvalidConversation("The last message must be from the user.")

    return [{"role": "system", "content": SYSTEM_PROMPT}, *messages]


def extract_answer(payload: Any) -> str:
    if not isinstance(payload, dict):
        raise ValueError("The model returned an invalid response.")

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("The model returned no answer.")

    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise ValueError("The model returned an invalid answer.")

    message = first_choice.get("message")
    if not isinstance(message, dict):
        raise ValueError("The model returned an invalid answer.")

    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("The model returned an empty answer.")

    return content.strip()
