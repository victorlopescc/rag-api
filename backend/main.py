import logging
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from routers import admin, admin_analytics, documents, query, webhook, users

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

app = FastAPI(
    title="RAG — Coordenação de Computação",
    version="1.0.0",
    redirect_slashes=False,
)

# Origens permitidas pelo CORS. Inclui:
#   - localhost:5173 (dev local com Vite)
#   - bot.vlopinhos.dev (produção, piloto na coordenação)
# Pra adicionar outros ambientes, edite a lista direto aqui — não vale
# a pena ler de env var (uma string mal formatada quebra silenciosamente
# e ninguém percebe até alguém abrir o painel).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://bot.vlopinhos.dev",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(users.router)
app.include_router(query.router)
app.include_router(webhook.router)
app.include_router(admin.router)
app.include_router(admin_analytics.router)


@app.on_event("startup")
def _warmup_retrieval() -> None:
    """Pre-carrega o que pode ser caro em runtime:
    - cross-encoder (~120MB, 2-5s)
    - índice BM25 (lê todos os chunks do Chroma, <1s pra ~1k chunks)
    No-op nos componentes desabilitados via config.
    """
    from pipeline.reranker import warmup as warmup_reranker
    from pipeline.bm25_index import warmup as warmup_bm25
    warmup_reranker()
    warmup_bm25()


@app.get("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=settings.api_port, reload=True)