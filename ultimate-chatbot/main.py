
import os
import requests

from typing import Any
from dotenv import load_dotenv
from litellm import completion

load_dotenv()

API_KEYS: dict[str, str | None] = {
    "groq": os.getenv("GROQ_API_KEY") or None,
    "openai": os.getenv("OPENAI_API_KEY") or None,
    "anthropic": os.getenv("ANTHROPIC_API_KEY") or None,
    "gemini": os.getenv("GEMINI_API_KEY") or None,
}

PROVIDERS: list[str] = ["groq", "openai", "anthropic", "gemini"]

MODEL_URLS: dict[str, str] = {
    "groq": "https://api.groq.com/openai/v1/models",
    "openai": "https://api.openai.com/v1/models",
    "anthropic": "https://api.anthropic.com/v1/models",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models",
}

MAX_TOKENS: int = 512

EXCLUDED_MODELS: list[str] = [
    "embedding", "whisper", "moderation", "dall-e",
    "transcribe", "tts", "rerank", "realtime",
    "imagegen", "gemini-2.5-flash-lite",
]

MODEL_PRIORITY: dict[str, int] = {
    "gemini-3.5-flash-lite": 0,
    "flash-lite": 1,
    "nano": 1,
    "mini": 2,
    "haiku": 2,
    "flash": 3,
    "instant": 3,
    "3b": 4,
    "4b": 4,
    "8b": 5,
    "small": 5,
    "27b": 12,
    "32b": 14,
    "70b": 20,
    "sonnet": 15,
    "pro": 20,
    "opus": 25,
}


def get_models(provider: str, api_key: str) -> list[dict[str, Any]]:
    headers: dict[str, str] = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, Any] = {}

    if provider == "anthropic":
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
        params = {"limit": 1000}

    elif provider == "gemini":
        headers = {}
        params = {"key": api_key, "pageSize": 1000}

    url: str | None = MODEL_URLS[provider]
    models: list[dict[str, Any]] = []

    try:
        while url:
            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=30,
            )
            response.raise_for_status()
            data: dict[str, Any] = response.json()

            items: list[dict[str, Any]] = data.get(
                "data", data.get("models", [])
            )
            models.extend(items)

            if provider == "gemini":
                token: str | None = data.get("nextPageToken")
                url = MODEL_URLS[provider] if token else None
                params = {
                    "key": api_key,
                    "pageSize": 1000,
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
        print(f"{provider}: discovery failed ({status})")
        return []

def get_model_id(provider: str, model: dict[str, Any]) -> str:
    name: str = model.get("id") or model.get("name", "")
    return name.removeprefix("models/") if provider == "gemini" else name

def is_chat_model(provider: str, model: dict[str, Any]) -> bool:
    name: str = get_model_id(provider, model).lower()

    if not name or any(word in name for word in EXCLUDED_MODELS):
        return False

    if provider == "anthropic":
        return name.startswith("claude-")

    if provider == "gemini":
        return "generateContent" in model.get("supportedGenerationMethods", [])

    if provider == "openai":
        return name.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-"))

    return model.get("active", True) is not False

def get_model_score(provider: str, model: dict[str, Any]) -> int:
    name: str = get_model_id(provider, model).lower()

    scores: list[int] = [
        score for word, score in MODEL_PRIORITY.items()
        if word in name
    ]

    return min(scores) if scores else 50

def discover_models() -> dict[str, list[str]]:
    available: dict[str, list[str]] = {}

    for provider in PROVIDERS:
        api_key: str | None = API_KEYS[provider]

        if not api_key:
            continue

        models: list[dict[str, Any]] = get_models(provider, api_key)
        models = [m for m in models if is_chat_model(provider, m)]

        models.sort(key=lambda m: get_model_score(provider, m))

        if models:
            available[provider] = [
                get_model_id(provider, model) for model in models
            ]

            print(
                f"{provider.upper()}: {len(models)} models | "
                f"Selected: {available[provider][0]}"
            )

        else:
            print(f"{provider.upper()}: No available chat models")

    return available

def chat(
    history: list[dict[str, str]],
    available: dict[str, list[str]],
) -> bool:

    for provider in PROVIDERS:
        if provider not in available:
            continue

        model_name: str = available[provider][0]
        model: str = f"{provider}/{model_name}"

        try:
            response = completion(
                model=model,
                messages=history,
                api_key=API_KEYS[provider],
                max_tokens=MAX_TOKENS,
                timeout=60,
            )

            answer: Any = response.choices[0].message.content

            if isinstance(answer, list):
                answer = "\n".join(
                    str(part.get("text", ""))
                    if isinstance(part, dict) else str(part)
                    for part in answer
                )

            if not answer:
                raise ValueError("Empty response")

            answer = str(answer)

            history.append({
                "role": "assistant",
                "content": answer,
            })

            print(f"\n[{provider.upper()} | {model_name}]")
            print(answer)

            return True

        except Exception as error:
            message: str = str(error)
            api_key: str | None = API_KEYS[provider]

            if api_key:
                message = message.replace(api_key, "[REDACTED]")

            print(f"{provider.upper()} failed: {type(error).__name__}")
            print("Switching to next provider...")

    print("All providers failed.")
    return False

def main() -> None:
    print("\nUltimate AI Chatbot")

    available: dict[str, list[str]] = discover_models()

    if not available:
        print("No configured providers available.")
        return

    history: list[dict[str, str]] = [{
        "role": "system",
        "content": "You are a helpful and intelligent AI assistant.",
    }]

    print("\nChat started. Commands: /exit, /clear, /models\n")

    while True:
        try:
            message: str = input("You: ").strip()

            if not message:
                continue

            if message == "/exit":
                break

            if message == "/clear":
                history = history[:1]
                print("History cleared.")
                continue

            if message == "/models":
                for provider, models in available.items():
                    print(f"{provider.upper()}: {', '.join(models)}")
                continue

            history.append({"role": "user", "content": message})

            if not chat(history, available):
                history.pop()

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")

if __name__ == "__main__":
    main()
