"""
MiniRAG — retrieval BM25 puro, sin dependencias pesadas.

Por qué BM25 y no un vector store:
    - ChromaDB arrastra opentelemetry-otlp-grpc con un _pb2.py viejo
      incompatible con protobuf>=4.21 (crash en Streamlit Cloud).
    - sentence-transformers + torch pesan ~500 MB.
    - Para un corpus de 4 archivos (~80 chunks), BM25 entrega resultados
      equivalentes en calidad a un vector search, en memoria, al instante.

rank-bm25 es pure-Python, ~2 MB instalado, 0 deps transitivas pesadas.
"""

from __future__ import annotations

import re
import string
import uuid
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────────────────────
# Datos
# ─────────────────────────────────────────────────────────────
@dataclass
class RetrievedChunk:
    content: str
    source: str
    score: float
    chunk_id: str


# ─────────────────────────────────────────────────────────────
# Tokenización y chunking
# ─────────────────────────────────────────────────────────────
_PUNCT_RE = re.compile(rf"[{re.escape(string.punctuation)}¿¡«»“”]")
_SPANISH_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "al", "algo", "algunas", "algunos", "ante", "antes",
        "como", "con", "contra", "cual", "cuando", "de", "del",
        "desde", "donde", "durante", "e", "el", "ella", "ellas",
        "ellos", "en", "entre", "era", "erais", "eran", "eras",
        "eres", "es", "esa", "esas", "ese", "eso", "esos", "esta",
        "estaba", "estabais", "estaban", "estabas", "estad", "estada",
        "estadas", "estado", "estados", "estamos", "estan", "estando",
        "estar", "estara", "estaran", "estaras", "estaremos", "estas",
        "este", "esto", "estos", "estoy", "fue", "fueron", "fui",
        "fuimos", "ha", "habia", "habida", "habidas", "habido",
        "habidos", "habiendo", "habra", "habran", "habras", "habre",
        "habremos", "habria", "habrian", "han", "has", "hasta", "hay",
        "haya", "hayan", "hayas", "he", "la", "las", "le", "les", "lo",
        "los", "mas", "me", "mi", "mis", "mucho", "muchos", "muy",
        "nada", "ni", "no", "nos", "nosotros", "nuestra", "nuestras",
        "nuestro", "nuestros", "o", "os", "otra", "otras", "otro",
        "otros", "para", "pero", "poco", "por", "porque", "que",
        "quien", "quienes", "se", "sea", "sean", "sera", "seran",
        "si", "sido", "siendo", "sin", "sobre", "sois", "solo", "somos",
        "son", "soy", "su", "sus", "suya", "suyas", "suyo", "suyos",
        "tambien", "tanto", "te", "tenia", "tenian", "tenido", "tengo",
        "ti", "tiene", "tienen", "todo", "todos", "tu", "tus", "tuya",
        "tuyas", "tuyo", "tuyos", "un", "una", "uno", "unos", "vosotras",
        "vosotros", "vuestra", "vuestras", "vuestro", "vuestros", "y",
        "ya", "yo",
    }
)


def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punct + drop stopwords. Simple y suficiente para BM25."""
    lowered = _PUNCT_RE.sub(" ", text.lower())
    return [t for t in lowered.split() if t and t not in _SPANISH_STOPWORDS]


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
                    for i in range(0, len(sent), max_size):
                        chunks.append(sent[i : i + max_size])
                    buffer = ""
        if buffer:
            chunks.append(buffer)

    return chunks


# ─────────────────────────────────────────────────────────────
# MiniRAG — BM25 en memoria
# ─────────────────────────────────────────────────────────────
class MiniRAG:
    """
    Retrieval BM25 en memoria. Ingesta lazy en el constructor.
    API compatible con la versión ChromaDB: search(query, k) → list[RetrievedChunk].
    """

    def __init__(
        self,
        data_dir: str | Path = "data",
        **_: object,  # acepta y descarta persist_dir/collection_name (compat)
    ):
        self.data_dir = Path(data_dir)
        self._chunks: list[dict] = []          # [{id, content, source}]
        self._bm25: BM25Okapi | None = None
        self._ingest()

    # ── Ingesta ──────────────────────────────────────────────
    def _ingest(self) -> int:
        if not self.data_dir.exists():
            return 0

        for path in sorted(self.data_dir.iterdir()):
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            if path.suffix.lower() == ".md":
                text = re.sub(
                    r"^---\s*\n.*?\n---\s*\n",
                    "",
                    text,
                    count=1,
                    flags=re.DOTALL,
                )

            for i, chunk in enumerate(chunk_by_paragraphs(text, max_size=500)):
                self._chunks.append(
                    {
                        "id": f"{path.stem}-{i}-{uuid.uuid4().hex[:6]}",
                        "content": chunk,
                        "source": path.name,
                    }
                )

        if not self._chunks:
            return 0

        tokenized = [_tokenize(c["content"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized)
        return len(self._chunks)

    # ── Búsqueda ─────────────────────────────────────────────
    def search(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        if not self._bm25 or not self._chunks:
            return []

        tokens = _tokenize(query)
        if not tokens:
            return []

        scores = self._bm25.get_scores(tokens)
        # Top-k por score descendente
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Normalizar al rango [0,1] dividiendo por el máximo (solo para display).
        max_score = max(scores) if len(scores) else 1.0
        norm = max_score if max_score > 0 else 1.0

        results: list[RetrievedChunk] = []
        for idx in top_idx:
            if scores[idx] <= 0:
                continue  # descartar chunks sin overlap con la query
            c = self._chunks[idx]
            results.append(
                RetrievedChunk(
                    content=c["content"],
                    source=c["source"],
                    score=float(scores[idx] / norm),
                    chunk_id=c["id"],
                )
            )
        return results

    # ── Utilidades ───────────────────────────────────────────
    def count(self) -> int:
        return len(self._chunks)
