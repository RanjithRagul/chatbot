
from chatbot.config import SYSTEM_PROMPT
from chatbot.model_discovery import discover_models
from chatbot.llm_client import chat
from chatbot import llm_client
from router.router import answer_rag_query, route_query
from log.logger import logger


def main() -> None:
    logger.log_event(
        event="APPLICATION_START",
        output_data="Chatbot application started",
    )

    try:
        print("\nUltimate AI Chatbot")

        try:
            available: dict[str, list[str]] = discover_models()

            logger.log_event(
                event="MODEL_DISCOVERY",
                output_data=available,
            )

        except Exception as error:
            logger.log_event(
                event="MODEL_DISCOVERY_FAILED",
                error=f"{type(error).__name__}: {error}",
            )
            raise

        if not available:
            print("No configured providers available.")

            logger.log_event(
                event="NO_PROVIDERS_AVAILABLE",
                output_data="No configured providers available",
            )
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

                logger.log_event(
                    event="USER_INPUT",
                    input_data=message,
                )

                if message == "/exit":
                    logger.log_event(
                        event="EXIT_COMMAND",
                        output_data="User requested application exit",
                    )
                    break

                if message == "/clear":
                    history = history[:1]

                    llm_client.conversation_summary = ""
                    llm_client.summarized_count = 0

                    print("History cleared.")

                    logger.log_event(
                        event="CONVERSATION_CLEARED",
                        output_data="Conversation history cleared",
                    )
                    continue

                if message == "/models":
                    for provider, models in available.items():
                        print(f"\n{provider.upper()}")

                        for model in models:
                            print(f"  - {model}")

                    logger.log_event(
                        event="MODELS_DISPLAYED",
                        output_data=available,
                    )
                    continue

                history.append({
                    "role": "user",
                    "content": message,
                })

                destination: str = route_query(
                    query=message,
                    available=available,
                )

                logger.log_event(
                    event="QUERY_ROUTED",
                    input_data=message,
                    output_data=f"destination={destination}",
                )

                if destination == "rag":
                    try:
                        answer: str = answer_rag_query(
                            query=message,
                            available=available,
                            history=history,
                        )

                        history.append({
                            "role": "assistant",
                            "content": answer,
                        })

                        print("\n[LOCAL RAG]")
                        print(answer)

                        logger.log_event(
                            event="RAG_RESPONSE",
                            input_data=message,
                            output_data=answer,
                        )

                    except Exception as error:
                        history.pop()

                        print(
                            "\nLocal RAG failed: "
                            f"{type(error).__name__}: {error}"
                        )

                        logger.log_event(
                            event="RAG_FAILED",
                            input_data=message,
                            error=f"{type(error).__name__}: {error}",
                        )

                    continue

                if not chat(history, available):
                    history.pop()

                    logger.log_event(
                        event="CHAT_FAILED",
                        input_data=message,
                        error="All available models failed",
                    )

                else:
                    answer = history[-1]["content"]

                    logger.log_event(
                        event="CHAT_RESPONSE",
                        input_data=message,
                        output_data=answer,
                    )

            except KeyboardInterrupt:
                logger.log_event(
                    event="KEYBOARD_INTERRUPT",
                    output_data="Application interrupted by user",
                )
                break

    except Exception as error:
        logger.log_event(
            event="APPLICATION_FAILED",
            error=f"{type(error).__name__}: {error}",
        )
        raise

    finally:
        print("\nGoodbye!")

        logger.log_event(
            event="APPLICATION_END",
            output_data="Chatbot application stopped",
        )


if __name__ == "__main__":
    main()
