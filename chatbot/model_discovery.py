
import time
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

from log.logger import logger


def get_models(
    provider: str,
    api_key: str
) -> list[dict[str, Any]]:

    headers: dict[str, str] = {
        "Authorization": f"Bearer {api_key}"
    }

    params: dict[str, Any] = {}
    url: str | None = MODEL_URLS[provider]
    models: list[dict[str, Any]] = []
    page: int = 1

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

    logger.log_event(
        event="PROVIDER_DISCOVERY_START",
        input_data={"provider": provider},
    )

    try:
        while url:

            start_time: float = time.perf_counter()

            logger.log_event(
                event="PROVIDER_API_REQUEST",
                input_data={
                    "provider": provider,
                    "page": page,
                    "endpoint": MODEL_URLS[provider],
                },
            )

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

            duration_ms: float = round(
                (time.perf_counter() - start_time) * 1000,
                2,
            )

            logger.log_event(
                event="PROVIDER_API_RESPONSE",
                input_data={
                    "provider": provider,
                    "page": page,
                    "http_status": response.status_code,
                },
                output_data={
                    "models_received": len(items),
                    "total_models": len(models),
                    "duration_ms": duration_ms,
                },
            )

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

            page += 1

        logger.log_event(
            event="PROVIDER_DISCOVERY_COMPLETE",
            input_data={"provider": provider},
            output_data={"total_models": len(models)},
        )

        return models

    except requests.RequestException as error:

        status: int | None = (
            error.response.status_code
            if error.response is not None else None
        )

        logger.log_event(
            event="PROVIDER_DISCOVERY_FAILED",
            input_data={
                "provider": provider,
                "http_status": status,
            },
            error=f"{type(error).__name__}: {error}",
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

    discovery_start: float = time.perf_counter()

    available: dict[str, list[str]] = {}

    logger.log_event(
        event="MODEL_DISCOVERY_START",
        input_data={"providers": PROVIDERS},
    )

    for provider in PROVIDERS:

        api_key: str | None = API_KEYS[provider]

        if not api_key:

            print(
                f"{provider.upper()}: API key not configured"
            )

            logger.log_event(
                event="PROVIDER_SKIPPED",
                input_data={"provider": provider},
                output_data="API key not configured",
            )

            continue

        models: list[dict[str, Any]] = get_models(
            provider,
            api_key
        )

        total_models: int = len(models)

        models = [
            model
            for model in models
            if is_chat_model(provider, model)
        ]

        filtered_models: int = len(models)

        logger.log_event(
            event="MODEL_FILTERING",
            input_data={
                "provider": provider,
                "total_models": total_models,
            },
            output_data={
                "chat_models": filtered_models,
                "excluded_models": total_models - filtered_models,
            },
        )

        models.sort(
            key=lambda model: get_model_score(provider, model)
        )

        if not models:

            print(
                f"{provider.upper()}: No available chat models"
            )

            logger.log_event(
                event="PROVIDER_NO_MODELS",
                input_data={"provider": provider},
                output_data="No available chat models",
            )

            continue

        available[provider] = [
            get_model_id(provider, model)
            for model in models
        ]

        logger.log_event(
            event="MODEL_PRIORITY_SORT",
            input_data={
                "provider": provider,
                "total_models": filtered_models,
            },
            output_data={
                "preferred_model": available[provider][0],
                "available_models": available[provider],
            },
        )

        print(
            f"{provider.upper()}: {len(models)} models | "
            f"Preferred: {available[provider][0]}"
        )

    duration_ms: float = round(
        (time.perf_counter() - discovery_start) * 1000,
        2,
    )

    logger.log_event(
        event="MODEL_DISCOVERY_COMPLETE",
        output_data={
            "available_providers": list(available.keys()),
            "total_providers": len(available),
            "models": available,
            "duration_ms": duration_ms,
        },
    )

    return available
