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

    # LLM (provider externo via litellm). O nome do modelo segue o
    # padrão "<provider>/<model>" do litellm — pra trocar de provider
    # (Groq, OpenAI, Anthropic, OpenRouter, ...) basta mudar a string
    # e setar a API key correspondente no .env.
    gemini_api_key: str = ""
    llm_model: str = "gemini/gemini-2.5-flash"
    embed_model: str = "gemini/gemini-embedding-001"

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
    # 15 chunks no prompt. Antes era 10 (limite confortável pro Qwen 7B
    # local), mas Gemini 3 Flash tem context window grande e consegue
    # filtrar ruído sozinho. Mais chunks ajuda especialmente em queries
    # de LISTA (ex.: "quais disciplinas no 5º período") onde o
    # reranker pode descartar borderline relevantes.
    max_chunks_retrieved: int = 15

    # API
    api_secret_key: str = "dev-secret"
    api_port: int = 8000

    # Retrieval híbrido: além da busca densa (embeddings), rodamos BM25
    # em cima dos chunks indexados e fundimos os rankings via RRF.
    # Pega match literal de nomes próprios, siglas e números de artigo
    # — onde embeddings perdem força. Doc-agnóstico.
    enable_bm25: bool = True
    # Profundidade da busca BM25 antes da fusão. Igual ao denso por
    # default — RRF normaliza por POSIÇÃO, não por score, então
    # tamanhos balanceados dão mais peso uniforme aos dois sinais.
    bm25_top_k: int = 50

    # Reranker = LLM-as-reranker via Gemini (mesmo modelo do RAG). Antes
    # rodávamos um cross-encoder local (mmarco-mMiniLMv2), que penalizava
    # chunks tabulares por desconhecimento do formato. O Gemini avalia
    # relevância semântica diretamente — caro? Não: ~$0.001/query.
    enable_reranker: bool = True
    # K de chunks que entram no reranker. 30 é menor que os 50 antigos
    # porque cada chunk vai no prompt do LLM (mais chunks = prompt maior
    # = mais tokens/$ por call). 30 ainda dá rede larga sem inflar custo.
    reranker_input_k: int = 30
    # Threshold mínimo do score do reranker (após sigmoid → [0,1]).
    # 0.0 = sem cutoff: passamos os top-k do reranker pro LLM sem
    # filtrar. Apostamos no Gemini 3 Flash pra ignorar ruído e dizer
    # "não encontrei" quando os chunks não respondem (ele é bom nisso,
    # ao contrário do Qwen 7B antigo que tendia a misturar info).
    # Histórico: 0.05 cortava chunks tabulares legítimos (cross-encoder
    # mmarco não entende formato "Per: X; Disciplina: Y;"). 0.02 ainda
    # cortava casos borderline. 0.0 entrega mais chunks → mais respostas.
    reranker_min_score: float = 0.0

    # Evolution API
    evolution_api_url: str = "http://localhost:8080"
    evolution_api_key: str = "evolution_key_troque"
    evolution_instance: str = "coordenacao"

    # Número do WhatsApp do bot, sem +, com DDI. Usado pra montar o link
    # ``wa.me/<numero>?text=...`` no fluxo de cadastro "aluno inicia a
    # conversa" — necessário pra não cair na detecção de spam do Meta
    # quando o bot envia mensagem proativamente. Ex: 5531990899055.
    bot_phone_number: str = ""

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
