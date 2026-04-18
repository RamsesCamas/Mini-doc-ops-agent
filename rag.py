"""
MiniRAG — RAG ligero con ChromaDB + embedding default (onnx + MiniLM-L6-v2).

Sin torch, sin sentence-transformers, sin langchain. Diseñado para caber en
el free tier de Streamlit Community Cloud (1 GB RAM).
"""

from __future__ import annotations

import os

# Mismo workaround que streamlit_app.py — por si rag.py se importa directo.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import re
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb


@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    chunk_id: str


def chunk_by_paragraphs(text: str, max_size: int = 500) -> list[str]:
    """
    Divide el texto por párrafos. Si un párrafo excede max_size, lo parte por
    oraciones. Si una oración sigue siendo demasiado larga, hace un corte duro.
    """
    chunks: list[str] = []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)
            continue

        # Partir por oraciones
        sentences = re.split(r"(?<=[.!?])\s+", para)
        buffer = ""
        for sent in sentences:
            if len(buffer) + len(sent) + 1 <= max_size:
                buffer = f"{buffer} {sent}".strip()
            else:
                if buffer:
                    chunks.append(buffer)
                if len(sent) <= max_size:
                    buffer = sent
                else:
                    # corte duro para oraciones muy largas
                    for i in range(0, len(sent), max_size):
                        chunks.append(sent[i : i + max_size])
                    buffer = ""
        if buffer:
            chunks.append(buffer)

    return chunks


class MiniRAG:
    """
    Vector store sobre ChromaDB embebido con la embedding function default
    (onnxruntime + all-MiniLM-L6-v2). Ingesta lazy desde un directorio de
    archivos .txt y .md.
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        persist_dir: str | Path = "chroma_mini",
        collection_name: str = "docops_mini",
    ):
        self.data_dir = Path(data_dir)
        self.persist_dir = Path(persist_dir)

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            # Sin embedding_function → usa default (onnx + MiniLM-L6-v2).
        )

        if self._collection.count() == 0:
            self._ingest()

    # ── Ingesta ──────────────────────────────────────────────
    def _ingest(self) -> int:
        if not self.data_dir.exists():
            return 0

        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict] = []

        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".md":
                # Remover frontmatter YAML
                text = re.sub(
                    r"^---\s*\n.*?\n---\s*\n",
                    "",
                    text,
                    count=1,
                    flags=re.DOTALL,
                )

            for i, chunk in enumerate(chunk_by_paragraphs(text, max_size=500)):
                ids.append(f"{path.stem}-{i}-{uuid.uuid4().hex[:6]}")
                docs.append(chunk)
                metas.append({"source": path.name, "chunk_index": i})

        if not docs:
            return 0

        # Chroma calcula embeddings internamente (default onnx).
        self._collection.add(ids=ids, documents=docs, metadatas=metas)
        return len(docs)

    # ── Búsqueda ─────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        res = self._collection.query(query_texts=[query], n_results=k)

        results: list[RetrievedChunk] = []
        if not res.get("ids") or not res["ids"][0]:
            return results

        for i in range(len(res["ids"][0])):
            distance = res["distances"][0][i]
            results.append(
                RetrievedChunk(
                    content=res["documents"][0][i],
                    source=(res["metadatas"][0][i] or {}).get("source", "?"),
                    score=1.0 - distance,  # cosine distance → similitud
                    chunk_id=res["ids"][0][i],
                )
            )
        return results

    # ── Utilidades ───────────────────────────────────────────
    def count(self) -> int:
        return self._collection.count()
