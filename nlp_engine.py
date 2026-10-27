from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

import faiss
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class DocumentProcessingError(Exception):
    """Raised when uploaded documents cannot be processed."""


@dataclass
class TextChunk:
    text: str
    source: str
    page: int


class SemanticSearchEngine:
    """Extract PDF text, create embeddings, and search chunks with FAISS."""

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._index = None
        self._chunks: list[TextChunk] = []
        self._source_names: list[str] = []

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def source_names(self) -> list[str]:
        return list(self._source_names)

    def clear(self) -> None:
        self._index = None
        self._chunks = []
        self._source_names = []

    @staticmethod
    def _clean_text(text: str) -> str:
        text = text.replace("\x00", " ")
        text = re.sub(r"-\s*\n\s*", "", text)  # join words split by PDF line wrapping
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _chunk_words(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
        if chunk_size <= 0:
            raise ValueError("Chunk size must be greater than zero.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("Overlap must be non-negative and smaller than chunk size.")
        words = text.split()
        if not words:
            return []
        chunks = []
        step = chunk_size - overlap
        for start in range(0, len(words), step):
            part = words[start:start + chunk_size]
            if part:
                chunks.append(" ".join(part))
            if start + chunk_size >= len(words):
                break
        return chunks

    @classmethod
    def _extract_chunks(cls, filename: str, pdf_bytes: bytes, chunk_size: int, overlap: int) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        try:
            with fitz.open(stream=pdf_bytes, filetype="pdf") as document:
                for page_number, page in enumerate(document, start=1):
                    text = cls._clean_text(page.get_text("text"))
                    for part in cls._chunk_words(text, chunk_size=chunk_size, overlap=overlap):
                        chunks.append(TextChunk(text=part, source=filename, page=page_number))
        except Exception as exc:
            raise DocumentProcessingError(f"Could not read '{filename}'. Ensure it is a valid, unencrypted PDF. Details: {exc}") from exc
        return chunks

    def add_pdfs(self, pdfs: Iterable[tuple[str, bytes]], chunk_size: int = 150, overlap: int = 30) -> int:
        """Replace the current index with chunks extracted from the supplied PDFs."""
        all_chunks: list[TextChunk] = []
        names: list[str] = []
        errors: list[str] = []

        for filename, content in pdfs:
            names.append(filename)
            try:
                all_chunks.extend(self._extract_chunks(filename, content, chunk_size, overlap))
            except DocumentProcessingError as exc:
                errors.append(str(exc))

        if errors and not all_chunks:
            raise DocumentProcessingError("\n".join(errors))
        if not all_chunks:
            self.clear()
            self._source_names = names
            return 0

        texts = [chunk.text for chunk in all_chunks]
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype=np.float32)
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        # Commit the new index only after extraction and embedding succeeded.
        self._chunks = all_chunks
        self._index = index
        self._source_names = names
        if errors:
            # The UI can still use valid PDFs; invalid ones are omitted from the index.
            self._last_warnings = errors
        else:
            self._last_warnings = []
        return len(all_chunks)

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not query or not query.strip():
            return []
        if self._index is None or not self._chunks:
            return []
        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.model.encode(
            [query.strip()],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32)
        k = min(int(top_k), len(self._chunks))
        scores, indices = self._index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[int(idx)]
            results.append({
                "text": chunk.text,
                "source": chunk.source,
                "page": chunk.page,
                "score": float(score),
            })
        return results
