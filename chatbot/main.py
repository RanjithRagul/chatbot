
from chatbot.config import SYSTEM_PROMPT
from chatbot.model_discovery import discover_models
from chatbot.llm_client import chat
from chatbot import llm_client


def main() -> None:

    print("\nUltimate AI Chatbot")

    available: dict[str, list[str]] = discover_models()

    if not available:
        print("No configured providers available.")
        return

    history: list[dict[str, str]] = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    print("\nChat started.")
    print("Commands: /exit, /clear, /models\n")

    while True:

        try:
            message: str = input("You: ").strip()

            if not message:
                continue

            if message == "/exit":
                break

            if message == "/clear":

                history = history[:1]

                llm_client.conversation_summary = ""
                llm_client.summarized_count = 0

                print("History cleared.")
                continue

            if message == "/models":

                for provider, models in available.items():

                    print(f"\n{provider.upper()}")

                    for model in models:
                        print(f"  - {model}")

                continue

            history.append({
                "role": "user",
                "content": message,
            })

            if not chat(history, available):
                history.pop()

        except KeyboardInterrupt:
            break

    print("\nGoodbye!")


if __name__ == "__main__":
    main()
