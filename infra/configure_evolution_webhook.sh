#!/usr/bin/env bash
# =============================================================================
# Configura o webhook da instância Evolution API com os eventos NECESSÁRIOS:
#
#   MESSAGES_UPSERT  — mensagens recebidas (perguntas do aluno)
#   MESSAGES_UPDATE  — VOTOS DE POLL (sem isso o clique nas enquetes não chega)
#   CONTACTS_UPSERT  — contatos novos (resolução de LID)
#   CONTACTS_UPDATE  — atualização de contatos (LID)
#
# Quando você troca a URL do ngrok ou recria a instância, rode esse script.
#
# Variáveis necessárias (lidas do infra/.env, ou exporta no shell):
#   EVOLUTION_API_URL        (default http://localhost:8080)
#   EVOLUTION_API_KEY        (sem default — obrigatório)
#   EVOLUTION_INSTANCE       (default 'coordenacao')
#   WEBHOOK_URL              (URL pública do backend, ex: https://xxx.ngrok-free.app/webhook)
#
# Uso:
#   bash infra/configure_evolution_webhook.sh https://abcd-12345.ngrok-free.app/webhook
# =============================================================================
set -euo pipefail

# Lê o .env se existir
if [[ -f "infra/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source infra/.env
    set +a
fi

API_URL="${EVOLUTION_API_URL:-http://localhost:8080}"
API_KEY="${EVOLUTION_API_KEY:?EVOLUTION_API_KEY não definido}"
INSTANCE="${EVOLUTION_INSTANCE:-coordenacao}"
WEBHOOK_URL="${1:-${WEBHOOK_URL:-}}"

if [[ -z "$WEBHOOK_URL" ]]; then
    echo "Uso: bash infra/configure_evolution_webhook.sh <URL_DO_WEBHOOK>"
    echo "Exemplo: bash infra/configure_evolution_webhook.sh https://abcd.ngrok-free.app/webhook"
    exit 2
fi

echo "→ Verificando config atual…"
curl -s "$API_URL/webhook/find/$INSTANCE" -H "apikey: $API_KEY" || true
echo ""

echo "→ Atualizando webhook → $WEBHOOK_URL"
curl -sS -X POST "$API_URL/webhook/set/$INSTANCE" \
    -H "apikey: $API_KEY" \
    -H "Content-Type: application/json" \
    -d "{
        \"enabled\": true,
        \"url\": \"$WEBHOOK_URL\",
        \"events\": [
            \"MESSAGES_UPSERT\",
            \"MESSAGES_UPDATE\",
            \"CONTACTS_UPSERT\",
            \"CONTACTS_UPDATE\"
        ]
    }"
echo ""
echo ""
echo "✓ Pronto. Confirme com: curl $API_URL/webhook/find/$INSTANCE -H 'apikey: ...'"
