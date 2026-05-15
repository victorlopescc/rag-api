"""Cache LRU thread-safe para deduplicar eventos de webhook.

A Evolution v1.8.7 reemite `messages.upsert` depois de enriquecer o pushName.
Sem esta proteção o bot responde duas vezes à mesma mensagem.
"""
from collections import OrderedDict
from threading import Lock


class MessageDedup:
    def __init__(self, max_size: int = 512) -> None:
        self._seen: OrderedDict[str, None] = OrderedDict()
        self._lock = Lock()
        self._max = max_size

    def seen(self, message_id: str) -> bool:
        """Retorna True se já vimos este id; caso contrário registra e retorna False."""
        if not message_id:
            return False
        with self._lock:
            if message_id in self._seen:
                return True
            self._seen[message_id] = None
            while len(self._seen) > self._max:
                self._seen.popitem(last=False)
        return False


message_dedup = MessageDedup()
