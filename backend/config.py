from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parent
ENV_FILE = BACKEND_DIR.parent / "infra" / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL
    postgres_user: str = "raguser"
    postgres_password: str = "ragpass"
    postgres_db: str = "ragdb"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "qwen2.5:7b-instruct-q4_K_M"
    ollama_embed_model: str = "nomic-embed-text"

    # ChromaDB — modo HTTP (Docker). Quando ``chroma_host`` está vazio,
    # caímos no modo embedded usando ``chroma_persist_path`` como
    # fallback (útil pra rodar testes ou setup de dev sem o container).
    chroma_host: str = "localhost"
    chroma_port: int = 8200
    chroma_persist_path: str = "./chroma_data"  # usado só no modo embedded

    # RAG
    chunk_size: int = 500
    chunk_overlap: int = 80
    # 0.20 dá margem pra perguntas curtas / com siglas (ex: "ADA")
    # contra chunks longos da PUC. 0.30 cortava demais com nomic-embed-text.
    similarity_threshold: float = 0.20
    # 10 chunks no prompt. Cobre casos onde o chunk crítico (ex.: a
    # cláusula "5 (cinco) pontos" do §4.2 da Resolução ADA) tem score
    # base baixo e só entra no top quando a janela é mais larga.
    # Em dev com IDE+Chrome aberto pode estourar OOM no qwen2.5:14b
    # (o user pode override via MAX_CHUNKS_RETRIEVED=8 no .env).
    max_chunks_retrieved: int = 10

    # API
    api_secret_key: str = "dev-secret"
    api_port: int = 8000

    # Reranker (cross-encoder local). Roda em CPU, sem API externa.
    # mmarco-mMiniLM é multilíngue (PT-BR ok), ~120MB no disco, ~500MB
    # de RAM em runtime. Se ``enable_reranker`` for False, o pipeline
    # cai pro ranking antigo (embedding + boosts).
    # Retrieval híbrido: além da busca densa (embeddings), rodamos BM25
    # em cima dos chunks indexados e fundimos os rankings via RRF.
    # Pega match literal de nomes próprios, siglas e números de artigo
    # — onde embeddings perdem força. Doc-agnóstico.
    enable_bm25: bool = True
    # Profundidade da busca BM25 antes da fusão. Igual ao denso por
    # default — RRF normaliza por POSIÇÃO, não por score, então
    # tamanhos balanceados dão mais peso uniforme aos dois sinais.
    bm25_top_k: int = 50

    enable_reranker: bool = True
    reranker_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Device do reranker. ``cpu`` é o default seguro: roda fora da GPU
    # e não compete por VRAM com o qwen14b servido pelo Ollama
    # (causa típica de "llama runner process has terminated: CUDA error"
    # quando os dois tentam dividir a memória de placa). Coloque ``cuda``
    # só se você tiver GPU dedicada com folga (>=24GB) e quiser ganhar
    # ~1s de latência por query.
    reranker_device: str = "cpu"
    # K de chunks que entram no reranker. Com retrieval híbrido (BM25
    # + denso fundidos por RRF), 50 dá ao reranker um pool maior pra
    # decidir o top-10 final. Latência sobe ~1s por query em CPU mas
    # aumenta a chance de o chunk certo estar disponível pro reranker.
    reranker_input_k: int = 50
    # Threshold mínimo do score do reranker (após sigmoid → [0,1]).
    # 0.05 corta só chunks claramente irrelevantes — o cross-encoder
    # tipicamente atribui >0.1 mesmo a chunks "borderline relevantes"
    # que o LLM consegue extrair informação útil. Limiares mais altos
    # (testamos 0.15) causaram queda em ~7 respostas válidas só pra
    # evitar 1 caso de hallucination — trade-off ruim.
    # A defesa contra hallucination ficou no PROMPT do LLM, que recusa
    # quando os chunks não contêm a informação.
    reranker_min_score: float = 0.05

    # Evolution API
    evolution_api_url: str = "http://localhost:8080"
    evolution_api_key: str = "evolution_key_troque"
    evolution_instance: str = "coordenacao"

    # Mongo da Evolution API (usado para resolver LID via ACKs de entrega)
    evolution_mongo_uri: str = "mongodb://localhost:27017"
    evolution_mongo_db: str = "evolution-whatsapp-api"

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    def model_post_init(self, __context) -> None:
        p = Path(self.chroma_persist_path)
        if not p.is_absolute():
            object.__setattr__(
                self,
                "chroma_persist_path",
                str((BACKEND_DIR / p).resolve()),
            )


settings = Settings()
