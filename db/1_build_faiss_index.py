
from pathlib import Path
from typing import Any
import json
import re

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer


class FaissIndexBuilder:

    def __init__(self) -> None:
        self.base_directory: Path = Path(__file__).resolve().parent.parent
        self.txt_folder: Path = self.base_directory / "pdf" / "extracted_logs"
        self.index_folder: Path = self.base_directory / "db" / "faiss_index"

        self.index_path: Path = self.index_folder / "index.faiss"
        self.metadata_path: Path = self.index_folder / "metadata.json"
        self.manifest_path: Path = self.index_folder / "manifest.json"

        self.gpu_count: int = torch.cuda.device_count()
        self.use_gpu: bool = torch.cuda.is_available() and self.gpu_count > 0
        self.devices: list[str] = (
            [f"cuda:{i}" for i in range(self.gpu_count)]
            if self.use_gpu else ["cpu"]
        )

        self.embedding_model: SentenceTransformer = SentenceTransformer(
            "BAAI/bge-small-en-v1.5",
            device=self.devices[0]
        )

        self.embedding_pool: Any = None

        if self.use_gpu and self.gpu_count > 1:
            self.embedding_pool = self.embedding_model.start_multi_process_pool(
                target_devices=self.devices
            )

        print(f"Execution device: {'GPU' if self.use_gpu else 'CPU'}")

        if self.use_gpu:
            for device in self.devices:
                device_id: int = int(device.split(":")[1])
                print(f"Using {device}: {torch.cuda.get_device_name(device_id)}")
        else:
            print("No CUDA GPU available. Using CPU.")

    def get_source_files(self) -> list[Path]:
        if not self.txt_folder.exists():
            raise FileNotFoundError("Extracted text folder not found.")

        return sorted(self.txt_folder.glob("*.txt"))

    def get_manifest(self) -> list[dict[str, Any]]:
        return [
            {
                "name": file.name,
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime_ns
            }
            for file in self.get_source_files()
        ]

    def split_pages(self, text: str) -> list[tuple[int, str]]:
        matches: list[re.Match[str]] = list(
            re.finditer(r"\[\[PAGE:(\d+)\]\]", text)
        )

        if not matches:
            return [(1, text)] if text.strip() else []

        pages: list[tuple[int, str]] = []

        for index, match in enumerate(matches):
            start: int = match.end()
            end: int = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(text)
            )

            page_text: str = text[start:end].strip()

            if page_text:
                pages.append((int(match.group(1)), page_text))

        return pages

    def chunk_text(
        self,
        text: str,
        chunk_size: int = 900,
        overlap: int = 150
    ) -> list[str]:

        if chunk_size <= 0 or not 0 <= overlap < chunk_size:
            raise ValueError("Invalid chunk size or overlap.")

        words: list[str] = text.split()
        chunks: list[str] = []
        start: int = 0

        while start < len(words):
            end: int = start
            length: int = 0

            while end < len(words):
                word_length: int = len(words[end])
                next_length: int = length + word_length + (1 if end > start else 0)

                if next_length > chunk_size and end > start:
                    break

                length = next_length
                end += 1

            chunks.append(" ".join(words[start:end]))

            if end >= len(words):
                break

            next_start: int = end
            overlap_length: int = 0

            while next_start > start:
                word_length: int = len(words[next_start - 1]) + 1

                if overlap_length + word_length > overlap:
                    break

                overlap_length += word_length
                next_start -= 1

            start = max(next_start, start + 1)

        return chunks

    def encode_documents(self, texts: list[str]) -> np.ndarray:

        if self.embedding_pool is not None:
            embeddings: np.ndarray = self.embedding_model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True,
                pool=self.embedding_pool
            )
        else:
            embeddings = self.embedding_model.encode(
                texts,
                batch_size=16,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=True
            )

        return np.ascontiguousarray(embeddings, dtype=np.float32)

    def build_index(self) -> None:

        source_files: list[Path] = self.get_source_files()

        if not source_files:
            raise FileNotFoundError("No extracted TXT files found.")

        texts: list[str] = []
        metadata: list[dict[str, Any]] = []

        for txt_file in source_files:
            print(f"Processing: {txt_file.name}")

            content: str = txt_file.read_text(encoding="utf-8")
            pages: list[tuple[int, str]] = self.split_pages(content)

            for page_number, page_text in pages:
                chunks: list[str] = self.chunk_text(page_text)

                for chunk in chunks:
                    texts.append(chunk)

                    metadata.append({
                        "source": txt_file.name,
                        "page": page_number,
                        "chunk_index": len(texts) - 1,
                        "text": chunk
                    })

        if not texts:
            raise ValueError("No text available for indexing.")

        print(f"Total documents: {len(source_files)}")
        print(f"Total chunks: {len(texts)}")
        print("Generating embeddings...")

        embeddings: np.ndarray = self.encode_documents(texts)

        self.index_folder.mkdir(parents=True, exist_ok=True)

        index: faiss.IndexFlatIP = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        index.add(embeddings)

        faiss.write_index(index, str(self.index_path))

        self.metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False),
            encoding="utf-8"
        )

        self.manifest_path.write_text(
            json.dumps(self.get_manifest(), indent=2),
            encoding="utf-8"
        )

        print("FAISS index created successfully.")
        print(f"Total indexed vectors: {index.ntotal}")
        print(f"Index location: {self.index_folder}")

    def close(self) -> None:

        if self.embedding_pool is not None:
            self.embedding_model.stop_multi_process_pool(
                self.embedding_pool
            )
            self.embedding_pool = None


if __name__ == "__main__":
    builder: FaissIndexBuilder = FaissIndexBuilder()

    try:
        builder.build_index()
    finally:
        builder.close()
