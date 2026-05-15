#!/usr/bin/env bash
# =============================================================================
# RESET TOTAL — destrói tudo e sobe do zero.
#
# Apaga:
#   - volumes dos containers (postgres_data, mongo_data, redis_data,
#     evolution_data) → zera ragdb, evolutiondb, contatos/chats do Evolution,
#     sessão do WhatsApp, cache do Redis
#   - diretório ./chroma_data (índice vetorial)
#
# Depois sobe os containers de novo. O init.sql do Postgres roda
# automaticamente e recria o schema (documents, students, qa_*, etc.)
# e o banco `evolutiondb`.
#
# Você ainda precisa:
#   1. reiniciar o backend uvicorn (pra limpar client Chroma em memória)
#   2. recriar a instância do WhatsApp na Evolution + escanear o QR
#
# Uso:
#   bash infra/reset_all.sh            # pede confirmação
#   bash infra/reset_all.sh --yes      # sem prompt
# =============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."

YES=0
for arg in "$@"; do
    case "$arg" in
        --yes|-y) YES=1 ;;
        *) echo "Argumento desconhecido: $arg"; exit 2 ;;
    esac
done

echo "============================================================"
echo "RESET TOTAL — isso é destrutivo e irreversível."
echo "============================================================"
echo "Vai apagar:"
echo "  • volumes: postgres_data, mongo_data, redis_data, evolution_data, chroma_data"
echo "  • diretório legado (modo embedded): ./backend/chroma_data"
echo ""

if [[ $YES -ne 1 ]]; then
    read -r -p "Confirma? digite 'apagar tudo' para prosseguir: " ans
    if [[ "$ans" != "apagar tudo" ]]; then
        echo "Abortado."
        exit 1
    fi
fi

echo ""
echo "→ Derrubando containers e removendo volumes…"
docker compose -f infra/docker-compose.yml down -v

echo ""
echo "→ Limpando diretório de persistência do ChromaDB…"
# settings.chroma_persist_path = ./chroma_data resolvido contra BACKEND_DIR,
# então o real é backend/chroma_data. Limpamos só esse — o ChromaDB recria
# os arquivos sozinho na próxima inicialização.
# Histórico: versões antigas do script criavam ./chroma_data na raiz por
# engano. Se aparecer de novo, pode apagar manualmente: `rmdir chroma_data`.
if [[ -d ./backend/chroma_data ]]; then
    echo "  rm -rf ./backend/chroma_data"
    rm -rf ./backend/chroma_data
fi

echo ""
echo "→ Subindo containers de novo…"
docker compose -f infra/docker-compose.yml up -d

echo ""
echo "→ Aguardando Postgres ficar saudável…"
for i in {1..30}; do
    if docker compose -f infra/docker-compose.yml exec -T postgres \
        pg_isready -U "${POSTGRES_USER:-raguser}" -d "${POSTGRES_DB:-ragdb}" \
        >/dev/null 2>&1; then
        echo "  ok"
        break
    fi
    sleep 1
done

echo ""
echo "✓ Feito. Próximos passos:"
echo "  1. reinicie o backend (uvicorn) pra descartar o ChromaClient em memória"
echo "  2. acesse sua rota HTTP pra recriar a instância + escanear o QR Code"
