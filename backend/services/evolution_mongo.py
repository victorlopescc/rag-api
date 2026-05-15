"""Leitura direta do Mongo da Evolution API.

A Evolution v1.8.7 grava os ACKs de entrega na coleção `messageUpdate` mas
não dispara o webhook `messages.update`. Consultar o Mongo é como
correlacionamos um LID ao aluno (via pending_welcome_id).
"""
import logging
from functools import lru_cache

from pymongo import MongoClient

from config import settings
from services.whatsapp import normalize_lid

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _db():
    client = MongoClient(settings.evolution_mongo_uri, serverSelectionTimeoutMS=3000)
    return client[settings.evolution_mongo_db]


def find_message_ids_for_lid(lid: str) -> list[str]:
    """key.id de mensagens nossas (fromMe:true) que receberam ACK deste LID."""
    local = normalize_lid(lid).split("@", 1)[0]
    if not local:
        return []
    try:
        cursor = _db().messageUpdate.find(
            {
                "fromMe": True,
                # Aceita tanto '<local>@lid' quanto '<local>:<device>@lid'.
                "remoteJid": {"$regex": f"^{local}(:[0-9]+)?@lid$"},
            },
            {"id": 1, "_id": 0},
        )
        return [doc["id"] for doc in cursor if doc.get("id")]
    except Exception as e:
        logger.error(f"Falha ao consultar messageUpdate no Mongo: {e}")
        return []
