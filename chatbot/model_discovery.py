
from typing import Any

import requests

from chatbot.config import (
    API_KEYS,
    PROVIDERS,
    MODEL_URLS,
    EXCLUDED_MODELS,
    MODEL_PRIORITY,
    DISCOVERY_TIMEOUT,
    MAX_DISCOVERY_MODELS,
)


def get_models(provider: str, api_key: str) -> list[dict[str, Any]]:

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}"
    }

    params: dict[str, Any] = {}
    url: str | None = MODEL_URLS[provider]
    models: list[dict[str, Any]] = []

    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }

        params = {
            "limit": MAX_DISCOVERY_MODELS
        }

    elif provider == "gemini":
        headers = {}

        params = {
            "key": api_key,
            "pageSize": MAX_DISCOVERY_MODELS,
        }

    try:
        while url:

            response: requests.Response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=DISCOVERY_TIMEOUT,
            )

            response.raise_for_status()

            data: dict[str, Any] = response.json()

            items: list[dict[str, Any]] = data.get(
                "data",
                data.get("models", [])
            )

            models.extend(items)

            if provider == "gemini":

                token: str | None = data.get("nextPageToken")

                url = MODEL_URLS[provider] if token else None

                params = {
                    "key": api_key,
                    "pageSize": MAX_DISCOVERY_MODELS,
                    "pageToken": token,
                } if token else {}

            elif provider == "openai":

                if data.get("has_more") and items:
                    params["after"] = items[-1]["id"]
                else:
                    url = None

            elif provider == "anthropic":

                last_id: str | None = data.get("last_id")

                if data.get("has_more") and last_id:
                    params["after_id"] = last_id
                else:
                    url = None

            else:
                url = None

        return models

    except requests.RequestException as error:

        status: int | None = (
            error.response.status_code
            if error.response is not None else None
        )

        print(
            f"{provider.upper()} discovery failed: {status}"
        )

        return []


def get_model_id(
    provider: str,
    model: dict[str, Any]
) -> str:

    name: str = model.get("id") or model.get("name", "")

    if provider == "gemini":
        return name.removeprefix("models/")

    return name


def is_chat_model(
    provider: str,
    model: dict[str, Any]
) -> bool:

    name: str = get_model_id(provider, model).lower()

    if not name:
        return False

    if any(word in name for word in EXCLUDED_MODELS):
        return False

    if provider == "anthropic":
        return name.startswith("claude-")

    if provider == "gemini":
        return (
            "generateContent"
            in model.get("supportedGenerationMethods", [])
        )

    if provider == "openai":
        return name.startswith(
            ("gpt-", "o1", "o3", "o4", "chatgpt-")
        )

    if provider == "groq":
        return model.get("active", True) is not False

    return False


def get_model_score(
    provider: str,
    model: dict[str, Any]
) -> int:

    name: str = get_model_id(provider, model).lower()

    scores: list[int] = [
        score
        for word, score in MODEL_PRIORITY.items()
        if word in name
    ]

    return min(scores) if scores else 50


def discover_models() -> dict[str, list[str]]:

    available: dict[str, list[str]] = {}

    for provider in PROVIDERS:

        api_key: str | None = API_KEYS[provider]

        if not api_key:
            print(
                f"{provider.upper()}: API key not configured"
            )
            continue

        models: list[dict[str, Any]] = get_models(
            provider,
            api_key
        )

        models = [
            model
            for model in models
            if is_chat_model(provider, model)
        ]

        models.sort(
            key=lambda model: get_model_score(provider, model)
        )

        if not models:
            print(
                f"{provider.upper()}: No available chat models"
            )
            continue

        available[provider] = [
            get_model_id(provider, model)
            for model in models
        ]

        print(
            f"{provider.upper()}: {len(models)} models | "
            f"Preferred: {available[provider][0]}"
        )

    return available
