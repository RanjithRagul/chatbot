
from pathlib import Path
from typing import Any
import json
import re
import time

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder

from log.logger import logger


class RerankRAG:

    def __init__(self) -> None:

        logger.log_event(
            event="RAG_INITIALIZATION_START",
            output_data="Initializing FAISS RAG pipeline",
        )

        try:
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

            embedding_start: float = time.perf_counter()

            self.embedding_model: SentenceTransformer = SentenceTransformer(
                "BAAI/bge-small-en-v1.5",
                device=self.devices[0]
            )

            logger.log_event(
                event="EMBEDDING_MODEL_LOADED",
                input_data="BAAI/bge-small-en-v1.5",
                output_data={
                    "device": self.devices[0],
                    "duration_ms": round(
                        (time.perf_counter() - embedding_start) * 1000, 2
                    ),
                },
            )

            reranker_start: float = time.perf_counter()

            self.reranker: CrossEncoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2",
                device=self.devices[0]
            )

            if self.use_gpu and self.gpu_count > 1:
                self.reranker.model = torch.nn.DataParallel(
                    self.reranker.model,
                    device_ids=list(range(self.gpu_count))
                )

            logger.log_event(
                event="RERANKER_MODEL_LOADED",
                input_data="cross-encoder/ms-marco-MiniLM-L-6-v2",
                output_data={
                    "device": self.devices[0],
                    "duration_ms": round(
                        (time.perf_counter() - reranker_start) * 1000, 2
                    ),
                },
            )

            self.index: Any = None
            self.metadata: list[dict[str, Any]] = []

            print(f"Execution device: {'GPU' if self.use_gpu else 'CPU'}")

            if self.use_gpu:
                for device in self.devices:
                    device_id: int = int(device.split(":")[1])
                    print(
                        f"Using {device}: "
                        f"{torch.cuda.get_device_name(device_id)}"
                    )
            else:
                print("No CUDA GPU available. Using CPU.")

            self.load_index()

            logger.log_event(
                event="RAG_INITIALIZATION_SUCCESS",
                output_data={
                    "device": self.devices,
                    "indexed_chunks": self.index.ntotal,
                },
            )

        except Exception as error:
            logger.log_event(
                event="RAG_INITIALIZATION_FAILED",
                error=f"{type(error).__name__}: {error}",
            )
            raise

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

    def load_index(self) -> None:

        logger.log_event(
            event="FAISS_INDEX_LOAD_START",
            input_data={
                "index_path": str(self.index_path),
                "metadata_path": str(self.metadata_path),
                "manifest_path": str(self.manifest_path),
            },
        )

        start_time: float = time.perf_counter()

        try:
            required_files: list[Path] = [
                self.index_path,
                self.metadata_path,
                self.manifest_path
            ]

            if not all(file.is_file() for file in required_files):
                raise FileNotFoundError(
                    "FAISS index not found. Run build_faiss_index.py first."
                )

            saved_manifest: list[dict[str, Any]] = json.loads(
                self.manifest_path.read_text(encoding="utf-8")
            )

            current_manifest: list[dict[str, Any]] = self.get_manifest()

            if saved_manifest != current_manifest:
                raise RuntimeError(
                    "Extracted documents have changed. "
                    "Run build_faiss_index.py to update the index."
                )

            self.index = faiss.read_index(str(self.index_path))

            self.metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )

            if self.index.ntotal != len(self.metadata):
                raise ValueError("FAISS index and metadata do not match.")

            if self.index.d != 384:
                raise ValueError("Unexpected embedding dimension.")

            duration_ms: float = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            print("FAISS index loaded successfully.")
            print(f"Total indexed chunks: {self.index.ntotal}")

            logger.log_event(
                event="FAISS_INDEX_LOAD_SUCCESS",
                input_data=str(self.index_path),
                output_data={
                    "total_chunks": self.index.ntotal,
                    "metadata_count": len(self.metadata),
                    "embedding_dimension": self.index.d,
                    "duration_ms": duration_ms,
                },
            )

        except Exception as error:
            logger.log_event(
                event="FAISS_INDEX_LOAD_FAILED",
                input_data=str(self.index_path),
                error=f"{type(error).__name__}: {error}",
            )
            raise

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

    def retrieve(
        self,
        query: str,
        top_k: int = 30
    ) -> list[dict[str, Any]]:

        logger.log_event(
            event="FAISS_RETRIEVAL_START",
            input_data={
                "query": query,
                "top_k": top_k,
            },
        )

        start_time: float = time.perf_counter()

        try:
            if not query.strip():
                raise ValueError("Query cannot be empty.")

            if top_k <= 0:
                raise ValueError("top_k must be positive.")

            embedding_start: float = time.perf_counter()

            query_embedding: np.ndarray = self.embedding_model.encode(
                [
                    f"Represent this sentence for searching relevant passages: {query}"
                ],
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            query_embedding = np.ascontiguousarray(
                query_embedding,
                dtype=np.float32
            )

            embedding_duration: float = round(
                (time.perf_counter() - embedding_start) * 1000, 2
            )

            logger.log_event(
                event="QUERY_EMBEDDING_COMPLETE",
                input_data=query,
                output_data={
                    "shape": str(query_embedding.shape),
                    "dimension": query_embedding.shape[1],
                    "duration_ms": embedding_duration,
                },
            )

            search_start: float = time.perf_counter()

            scores, indices = self.index.search(
                query_embedding,
                min(top_k, self.index.ntotal)
            )

            search_duration: float = round(
                (time.perf_counter() - search_start) * 1000, 2
            )

            candidates: list[dict[str, Any]] = []

            for score, index in zip(scores[0], indices[0]):
                if index < 0:
                    continue

                candidates.append({
                    "vector_score": float(score),
                    "document": self.metadata[index]
                })

            duration_ms: float = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            logger.log_event(
                event="FAISS_RETRIEVAL_COMPLETE",
                input_data={
                    "query": query,
                    "top_k": top_k,
                },
                output_data={
                    "candidates": len(candidates),
                    "embedding_duration_ms": embedding_duration,
                    "search_duration_ms": search_duration,
                    "total_duration_ms": duration_ms,
                    "results": [
                        {
                            "source": item["document"].get("source"),
                            "page": item["document"].get("page"),
                            "vector_score": item["vector_score"],
                        }
                        for item in candidates
                    ],
                },
            )

            return candidates

        except Exception as error:
            logger.log_event(
                event="FAISS_RETRIEVAL_FAILED",
                input_data=query,
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_n: int = 5
    ) -> list[dict[str, Any]]:

        logger.log_event(
            event="RERANKING_START",
            input_data={
                "query": query,
                "candidate_count": len(candidates),
                "top_n": top_n,
            },
        )

        start_time: float = time.perf_counter()

        try:
            if not candidates:
                logger.log_event(
                    event="RERANKING_COMPLETE",
                    output_data="No candidates available",
                )
                return []

            if top_n <= 0:
                raise ValueError("top_n must be positive.")

            pairs: list[tuple[str, str]] = [
                (query, item["document"]["text"])
                for item in candidates
            ]

            scores: np.ndarray = np.asarray(
                self.reranker.predict(
                    pairs,
                    batch_size=16,
                    show_progress_bar=False
                )
            )

            results: list[dict[str, Any]] = []

            for score, candidate in zip(scores, candidates):
                results.append({
                    "rerank_score": float(score),
                    "vector_score": candidate["vector_score"],
                    "document": candidate["document"]
                })

            results.sort(
                key=lambda item: item["rerank_score"],
                reverse=True
            )

            final_results: list[dict[str, Any]] = results[:top_n]

            duration_ms: float = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            logger.log_event(
                event="RERANKING_COMPLETE",
                input_data={
                    "candidates": len(candidates),
                    "top_n": top_n,
                },
                output_data={
                    "selected": len(final_results),
                    "duration_ms": duration_ms,
                    "results": [
                        {
                            "source": item["document"].get("source"),
                            "page": item["document"].get("page"),
                            "vector_score": item["vector_score"],
                            "rerank_score": item["rerank_score"],
                        }
                        for item in final_results
                    ],
                },
            )

            return final_results

        except Exception as error:
            logger.log_event(
                event="RERANKING_FAILED",
                input_data={
                    "query": query,
                    "candidate_count": len(candidates),
                },
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def get_document(self, source: str) -> dict[str, Any]:

        logger.log_event(
            event="DOCUMENT_READ_START",
            input_data=source,
        )

        try:
            file_path: Path = self.txt_folder / source

            if not file_path.is_file() or file_path.parent != self.txt_folder:
                raise FileNotFoundError("Document not found.")

            text: str = file_path.read_text(encoding="utf-8")

            pages: list[tuple[int, str]] = self.split_pages(text)

            document: str = "\n\n".join(
                f"[Page {page_number}]\n{page_text}"
                for page_number, page_text in pages
            )

            result: dict[str, Any] = {
                "source": source,
                "total_pages": len(pages),
                "document": document
            }

            logger.log_event(
                event="DOCUMENT_READ_COMPLETE",
                input_data=source,
                output_data={
                    "total_pages": len(pages),
                    "characters": len(document),
                },
            )

            return result

        except Exception as error:
            logger.log_event(
                event="DOCUMENT_READ_FAILED",
                input_data=source,
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def search(
        self,
        query: str,
        top_k: int = 30,
        top_n: int = 5
    ) -> list[dict[str, Any]]:

        logger.log_event(
            event="RAG_SEARCH_START",
            input_data={
                "query": query,
                "top_k": top_k,
                "top_n": top_n,
            },
        )

        start_time: float = time.perf_counter()

        try:
            candidates: list[dict[str, Any]] = self.retrieve(
                query,
                top_k
            )

            results: list[dict[str, Any]] = self.rerank(
                query,
                candidates,
                top_n
            )

            duration_ms: float = round(
                (time.perf_counter() - start_time) * 1000, 2
            )

            logger.log_event(
                event="RAG_SEARCH_COMPLETE",
                input_data=query,
                output_data={
                    "retrieved_candidates": len(candidates),
                    "final_results": len(results),
                    "duration_ms": duration_ms,
                },
            )

            return results

        except Exception as error:
            logger.log_event(
                event="RAG_SEARCH_FAILED",
                input_data=query,
                error=f"{type(error).__name__}: {error}",
            )
            raise

    def display_results(
        self,
        results: list[dict[str, Any]]
    ) -> None:

        if not results:
            print("No relevant results found.")
            return

        for position, result in enumerate(results, start=1):
            document: dict[str, Any] = result["document"]

            print(f"\nResult {position}")
            print(f"Source: {document['source']}")
            print(f"Page: {document['page']}")
            print(f"Chunk: {document['chunk_index']}")
            print(f"Rerank score: {result['rerank_score']:.4f}")
            print(f"Vector score: {result['vector_score']:.4f}")
            print(f"\n{document['text']}")
            print("-" * 70)


if __name__ == "__main__":

    rag: RerankRAG = RerankRAG()

    while True:
        query: str = input("\nEnter your question (or exit): ").strip()

        if query.lower() in {"exit", "quit"}:
            break

        if not query:
            continue

        results: list[dict[str, Any]] = rag.search(query)

        rag.display_results(results)
