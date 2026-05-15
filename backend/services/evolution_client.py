"""Cliente HTTP da Evolution API.

Concentra em um único lugar as chamadas ao endpoint /message/sendText.
O Evolution v1.8.7 só aceita números reais (@s.whatsapp.net) — enviar
para um LID cru devolve 400.
"""
import logging

import httpx

from config import settings

logger = logging.getLogger(__name__)


class EvolutionClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        instance: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = (base_url or settings.evolution_api_url).rstrip("/")
        self.api_key = api_key or settings.evolution_api_key
        self.instance = instance or settings.evolution_instance
        self.timeout = timeout

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json", "apikey": self.api_key}

    async def send_text(self, number: str, text: str) -> str | None:
        """Envia uma mensagem e retorna o key.id, ou None se falhou."""
        url = f"{self.base_url}/message/sendText/{self.instance}"
        payload = {"number": number, "textMessage": {"text": text}}

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers)

        if resp.status_code not in (200, 201):
            logger.error(f"Evolution API erro {resp.status_code}: {resp.text}")
            return None

        message_id = resp.json().get("key", {}).get("id")
        logger.info(f"Mensagem enviada para {number} (id={message_id})")
        return message_id

    async def send_poll(
        self, number: str, name: str, options: list[str],
        selectable_count: int = 1,
    ) -> str | None:
        """Envia uma enquete WhatsApp. Retorna o key.id para casar os votos.

        Rota: ``POST /message/sendPoll/{instance}`` (Evolution v1.8.x).
        Polls nativas funcionam em Baileys — diferente de buttons/lists que
        o WhatsApp depreciou. Ver doc.evolution-api.com.
        """
        url = f"{self.base_url}/message/sendPoll/{self.instance}"
        payload = {
            "number": number,
            "pollMessage": {
                "name": name,
                "selectableCount": selectable_count,
                "values": options,
            },
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, json=payload, headers=self._headers)

        if resp.status_code not in (200, 201):
            logger.error(f"Evolution API poll erro {resp.status_code}: {resp.text}")
            return None

        message_id = resp.json().get("key", {}).get("id")
        logger.info(f"Poll enviada para {number} (id={message_id})")
        return message_id


evolution_client = EvolutionClient()
