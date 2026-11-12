
from pathlib import Path
from typing import Dict, List
import time

import pymupdf

from log.logger import logger


class PDFTextExtractor:

    def __init__(self) -> None:

        self.base_directory: Path = Path(__file__).resolve().parent
        self.directory: Path = self.base_directory / "pdf_files"
        self.output_directory: Path = self.base_directory / "extracted_logs"

    def extract_all_pdfs(self) -> Dict[str, List[str]]:

        start_time: float = time.perf_counter()

        pdf_data: Dict[str, List[str]] = {}

        logger.log_event(
            event="PDF_EXTRACTION_START",
            input_data=str(self.directory),
        )

        try:

            if not self.directory.exists():

                logger.log_event(
                    event="PDF_DIRECTORY_CHECK",
                    input_data=str(self.directory),
                    output_data="PDF directory does not exist",
                )

                return pdf_data

            pdfs: List[Path] = sorted(
                path for path in self.directory.iterdir()
                if path.is_file() and path.suffix.lower() == ".pdf"
            )

            logger.log_event(
                event="PDF_DISCOVERY",
                input_data=str(self.directory),
                output_data={
                    "total_pdfs": len(pdfs),
                    "files": [pdf.name for pdf in pdfs],
                },
            )

            for pdf_path in pdfs:

                pdf_start: float = time.perf_counter()

                logger.log_event(
                    event="PDF_OPEN",
                    input_data=pdf_path.name,
                )

                try:

                    with pymupdf.open(pdf_path) as document:

                        pages: List[str] = []

                        for page_number, page in enumerate(
                            document,
                            start=1
                        ):

                            page_start: float = time.perf_counter()

                            text: str = page.get_text("text")

                            pages.append(text)

                            logger.log_event(
                                event="PDF_PAGE_EXTRACTION",
                                input_data={
                                    "file": pdf_path.name,
                                    "page": page_number,
                                },
                                output_data={
                                    "characters": len(text),
                                    "duration_ms": round(
                                        (time.perf_counter() - page_start) * 1000,
                                        2,
                                    ),
                                },
                            )

                        pdf_data[pdf_path.name] = pages

                    logger.log_event(
                        event="PDF_EXTRACTION_COMPLETE",
                        input_data=pdf_path.name,
                        output_data={
                            "total_pages": len(pages),
                            "total_characters": sum(
                                len(page) for page in pages
                            ),
                            "duration_ms": round(
                                (time.perf_counter() - pdf_start) * 1000,
                                2,
                            ),
                        },
                    )

                except Exception as error:

                    logger.log_event(
                        event="PDF_EXTRACTION_FAILED",
                        input_data=pdf_path.name,
                        error=f"{type(error).__name__}: {error}",
                    )

                    raise

            logger.log_event(
                event="PDF_PROCESSING_COMPLETE",
                output_data={
                    "total_pdfs": len(pdf_data),
                    "total_pages": sum(
                        len(pages) for pages in pdf_data.values()
                    ),
                    "duration_ms": round(
                        (time.perf_counter() - start_time) * 1000,
                        2,
                    ),
                },
            )

            return pdf_data

        except Exception as error:

            logger.log_event(
                event="PDF_EXTRACTION_FAILED",
                input_data=str(self.directory),
                error=f"{type(error).__name__}: {error}",
            )

            raise

    def save_extracted_text(
        self,
        pdf_data: Dict[str, List[str]]
    ) -> None:

        logger.log_event(
            event="PDF_SAVE_START",
            input_data={
                "total_pdfs": len(pdf_data),
                "output_directory": str(self.output_directory),
            },
        )

        try:

            self.output_directory.mkdir(
                parents=True,
                exist_ok=True
            )

            total_saved: int = 0

            for i, (pdf_name, pages) in enumerate(pdf_data.items()):

                file_name: str = f"{i}_{Path(pdf_name).stem}.txt"

                output_file: Path = (
                    self.output_directory / file_name
                )

                save_start: float = time.perf_counter()

                with output_file.open(
                    "w",
                    encoding="utf-8"
                ) as file:

                    file.write(
                        "\n".join(
                            f"[[PAGE:{page_number}]]\n{text}"
                            for page_number, text in enumerate(
                                pages,
                                start=1
                            )
                        )
                    )

                total_saved += 1

                logger.log_event(
                    event="PDF_SAVE_COMPLETE",
                    input_data={
                        "source": pdf_name,
                        "pages": len(pages),
                    },
                    output_data={
                        "output_file": str(output_file),
                        "size_bytes": output_file.stat().st_size,
                        "duration_ms": round(
                            (time.perf_counter() - save_start) * 1000,
                            2,
                        ),
                    },
                )

            logger.log_event(
                event="PDF_SAVE_SUMMARY",
                output_data={
                    "total_saved": total_saved,
                    "output_directory": str(self.output_directory),
                },
            )

        except Exception as error:

            logger.log_event(
                event="PDF_SAVE_FAILED",
                input_data=str(self.output_directory),
                error=f"{type(error).__name__}: {error}",
            )

            raise

    def remove_all_pdf(self) -> None:

        if not self.directory.exists():
            return

        for pdf_path in self.directory.iterdir():

            if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":

                pdf_path.unlink()

                logger.log_event(
                    event="PDF_FILE_REMOVED",
                    input_data=str(pdf_path),
                    output_data="PDF file deleted",
                )


extractor = PDFTextExtractor()

try:

    pdf_data: Dict[str, List[str]] = extractor.extract_all_pdfs()

    extractor.save_extracted_text(pdf_data)

except Exception as error:

    logger.log_event(
        event="PDF_PROCESSING_FAILED",
        error=f"{type(error).__name__}: {error}",
    )

    raise

# extractor.remove_all_pdf()
