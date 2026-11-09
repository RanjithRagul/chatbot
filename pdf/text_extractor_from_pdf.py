
from pathlib import Path
from typing import Dict, List
import pymupdf

class PDFTextExtractor:
    def __init__(self) -> None:
        self.base_directory: Path = Path(__file__).resolve().parent
        self.directory: Path = self.base_directory / "pdf_files"
        self.output_directory: Path = self.base_directory / "extracted_logs"
    def extract_all_pdfs(self) -> Dict[str, List[str]]:
        pdf_data: Dict[str, List[str]] = {}
        if not self.directory.exists():
            return pdf_data
        pdfs: List[Path] = sorted(
            path for path in self.directory.iterdir()
            if path.is_file() and path.suffix.lower() == ".pdf"
        )
        for pdf_path in pdfs:
            with pymupdf.open(pdf_path) as document:
                pdf_data[pdf_path.name] = [
                    page.get_text("text")
                    for page in document
                ]
        return pdf_data

    def save_extracted_text(self, pdf_data: Dict[str, List[str]]) -> None:
        self.output_directory.mkdir(parents=True, exist_ok=True)
        for i, (pdf_name, pages) in enumerate(pdf_data.items()):
            file_name: str = f"{i}_{Path(pdf_name).stem}.txt"
            output_file: Path = self.output_directory / file_name

            with output_file.open("w", encoding="utf-8") as file:
                file.write(
                    "\n".join(
                        f"[[PAGE:{page_number}]]\n{text}"
                        for page_number, text in enumerate(pages, start=1)
                    )
                )

    def remove_all_pdf(self) -> None:
        if not self.directory.exists():
            return
        for pdf_path in self.directory.iterdir():
            if pdf_path.is_file() and pdf_path.suffix.lower() == ".pdf":
                pdf_path.unlink()

extractor = PDFTextExtractor()
pdf_data: Dict[str, List[str]] = extractor.extract_all_pdfs()
extractor.save_extracted_text(pdf_data)

# extractor.remove_all_pdf()
