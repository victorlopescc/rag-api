#!/usr/bin/env bash
# =============================================================================
# Sobe o backend FastAPI em modo PRODUÇÃO.
#
# Diferenças do modo dev (uvicorn main:app --reload):
# - Múltiplos workers (4 por default) → vários processos atendem em paralelo.
#   Sem isso, uma pergunta no WhatsApp (RAG ~20s) trava todo o servidor e
#   nem o admin consegue carregar os GETs.
# - Sem --reload (que é incompatível com --workers e custa CPU à toa).
# - --proxy-headers — necessário se o backend ficar atrás de Nginx/Caddy.
# - --no-access-log silencia o spam de "POST /webhook 200 OK" (mantém os
#   logs do app que já mostram cada evento).
#
# Uso:
#   bash infra/run_backend.sh                  # 4 workers, porta 8000
#   WORKERS=2 bash infra/run_backend.sh        # quantos quiser
#   API_PORT=9000 bash infra/run_backend.sh    # outra porta
#
# Por que 4 workers e não 8?
# O Ollama processa 1 generate por vez (gargalo de GPU/CPU). Mais workers
# que isso só fariam fila no Ollama. Em uma máquina com mais GPU/CPU pra
# rodar mais Ollama em paralelo, aumente.
# =============================================================================
set -euo pipefail

# Ativa a venv se existir e não tiver sido ativada.
if [[ -d ".venv" && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/Scripts/activate 2>/dev/null || source .venv/bin/activate
fi

cd backend

WORKERS="${WORKERS:-4}"
PORT="${API_PORT:-8000}"
HOST="${HOST:-0.0.0.0}"

echo "→ Subindo backend: $WORKERS workers em $HOST:$PORT"
exec uvicorn main:app \
    --host "$HOST" \
    --port "$PORT" \
    --workers "$WORKERS" \
    --proxy-headers \
    --no-access-log
