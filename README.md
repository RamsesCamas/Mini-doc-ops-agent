# Mini DocOps Agent

Versión reducida del **DocOps Agent** (AI Engineer Bootcamp, Código Facilito — Clase 16), diseñada para correr en el **free tier de Streamlit Community Cloud** (1 GB RAM).

## Qué hace

Responde preguntas sobre documentos corporativos usando un RAG simple (retrieve → generate) con Groq, y protege la conversación con guardrails de entrada (anti prompt-injection) y salida (PII scrubbing).

## Stack

5 dependencias, sin frameworks pesados:

| Paquete | Uso |
|---|---|
| `streamlit` | UI del chat |
| `groq` | cliente LLM (free tier) |
| `rank-bm25` | retrieval puro-Python sobre los documentos |
| `tiktoken` | conteo de tokens |
| `python-dotenv` | carga de `.env` local |

**Huella estimada**: ~250 MB RAM. Sin `torch`, `sentence-transformers`, `langchain`, `langgraph`, `chromadb`.

**¿Por qué BM25 y no vector search?** ChromaDB arrastra `opentelemetry-otlp-grpc`
con un `_pb2.py` viejo que crashea con `protobuf>=4.21` en el sandbox de
Streamlit Cloud. Para un corpus de 4 archivos (~80 chunks), BM25 entrega
recall equivalente a un vector search — y cabe en 2 MB sin drama.

## Correr localmente

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="gsk_..."
streamlit run streamlit_app.py
```

El primer arranque tarda ~2 s (indexado BM25 en memoria del corpus). Las queries responden en ~1 s más el roundtrip a Groq.

## Deploy a Streamlit Community Cloud

1. Sube este directorio a un repo de GitHub (público o privado).
2. Entra a <https://share.streamlit.io> → **New app**.
3. Apunta al repo + rama + archivo `streamlit_app.py`.
4. En **Advanced settings → Secrets**, pega:
   ```toml
   GROQ_API_KEY = "gsk_..."
   GROQ_MODEL = "llama-3.3-70b-versatile"
   ```
5. **Deploy**. El primer build dura 3-5 min.

## Estructura

```
.
├── streamlit_app.py         # UI + loop del agente
├── rag.py                   # MiniRAG con BM25 en memoria
├── guardrails.py            # InputGuardrail + OutputGuardrail (+ ToolGuardrail)
├── requirements.txt
├── .streamlit/
│   ├── config.toml          # tema oscuro + headless
│   └── secrets.toml.example
└── data/                    # corpus indexado en arranque
    ├── manual_onboarding.txt
    ├── politica_vacaciones.txt
    ├── proceso_soporte_tecnico.txt
    └── langchain-readme.md
```

## Limitaciones conocidas

- **Sin persistencia**: todo el índice BM25 vive en memoria; se reconstruye en cada cold start (~2 s).
- **Solo retrieval léxico**: sin embeddings, BM25 matchea por palabras exactas + variantes morfológicas vía stemming implícito. Para queries con sinónimos muy diferentes al corpus, el recall puede sufrir. Para este corpus pequeño y bien estructurado, funciona de sobra.
- **Corpus estático**: solo los 4 archivos en `data/`. No hay upload en la UI.
- **Un solo call LLM**: no hay planner/verifier/rerank. Si necesitas revisión de calidad o control de iteraciones, usa la versión completa del bootcamp.
- **Groq TPD** free tier: ~100k tokens/día → ~50 queries antes de saltar el límite.

## Diferencias vs. versión completa del bootcamp

La versión completa incluye: multi-agente con LangGraph (Planner/Retriever/Executor/Verifier), retrieval híbrido (BM25 + vector + cross-encoder reranking), memoria persistente con checkpointing, Human-in-the-Loop, observabilidad con Arize Phoenix, evals tipo-RAGAS, MCP server, y configs de deploy para HF Spaces y Render. Si quieres todo eso → repo completo del bootcamp.

## Autor

**Ramsés Camas** — AI Engineer Bootcamp Código Facilito (2026).
