from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chatbot.config import TOPIC_REJECTION_MESSAGE
from chatbot.main import main
from pdf.text_extractor_from_pdf import PDFTextExtractor
from router import router


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RouterTests(unittest.TestCase):
    def test_routes_in_scope_general_question_to_chat(self) -> None:
        with patch.object(
            router,
            "generate",
            return_value="chat",
        ):
            destination = router.route_query(
                "What is Python?",
                {"groq": ["model"]},
            )

        self.assertEqual(destination, "chat")

    def test_routes_out_of_scope_request(self) -> None:
        with patch.object(
            router,
            "generate",
            return_value="out_of_scope",
        ) as generate:
            destination = router.route_query(
                "What is the weather?",
                {"groq": ["model"]},
            )

        self.assertEqual(destination, "out_of_scope")
        prompt = generate.call_args.kwargs["messages"][0]["content"]
        self.assertIn("Python programming", prompt)

    def test_routes_local_document_request_to_rag(self) -> None:
        with patch.object(router, "generate", return_value="rag") as generate:
            destination = router.route_query(
                "What does my PDF say about Gatsby?",
                {"groq": ["model"]},
            )

        self.assertEqual(destination, "rag")
        prompt = generate.call_args.kwargs["messages"][0]["content"]
        self.assertIn(
            "Requests to consult local documents remain in scope",
            prompt,
        )

    def test_router_failure_fails_closed_when_topic_boundary_is_enabled(self) -> None:
        with (
            patch.object(router, "generate", side_effect=RuntimeError("offline")),
            patch.object(router.logger, "log_event"),
            patch("builtins.print"),
        ):
            destination = router.route_query(
                "What is Python?",
                {"groq": ["model"]},
            )

        self.assertEqual(destination, "out_of_scope")


class ChatFlowTests(unittest.TestCase):
    def test_rejects_out_of_scope_request_without_calling_chat_model(self) -> None:
        output = io.StringIO()
        with (
            patch("chatbot.main.discover_models", return_value={"groq": ["model"]}),
            patch("chatbot.main.route_query", return_value="out_of_scope"),
            patch("chatbot.main.chat") as chat,
            patch("chatbot.main.logger.log_event"),
            patch("builtins.input", side_effect=["Tell me the weather", "/exit"]),
            contextlib.redirect_stdout(output),
        ):
            main()

        chat.assert_not_called()
        self.assertIn(TOPIC_REJECTION_MESSAGE, output.getvalue())


class PdfExtractorTests(unittest.TestCase):
    def test_import_does_not_extract_or_create_output_directory(self) -> None:
        source = PROJECT_ROOT / "pdf" / "text_extractor_from_pdf.py"

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            temporary_pdf_directory = temporary_root / "pdf"
            temporary_pdf_directory.mkdir()
            copied_module = temporary_pdf_directory / source.name
            shutil.copyfile(source, copied_module)

            spec = importlib.util.spec_from_file_location(
                "isolated_pdf_extractor",
                copied_module,
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)

            with patch(
                "pymupdf.open",
                side_effect=AssertionError("PDF opened on import"),
            ):
                spec.loader.exec_module(module)

            self.assertFalse(
                (temporary_pdf_directory / "extracted_logs").exists()
            )

    def test_main_extracts_and_saves_pages(self) -> None:
        with (
            patch.object(PDFTextExtractor, "extract_all_pdfs", return_value={"a.pdf": ["page"]}) as extract,
            patch.object(PDFTextExtractor, "save_extracted_text") as save,
        ):
            from pdf.text_extractor_from_pdf import main as extractor_main

            extractor_main()

        extract.assert_called_once_with()
        save.assert_called_once_with({"a.pdf": ["page"]})


if __name__ == "__main__":
    unittest.main()
