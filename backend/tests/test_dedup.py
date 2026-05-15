"""Testa o cache LRU de deduplicação de mensagens."""
from threading import Thread

from services.dedup import MessageDedup


def test_first_seen_returns_false():
    d = MessageDedup()
    assert d.seen("msg-1") is False


def test_second_seen_returns_true():
    d = MessageDedup()
    d.seen("msg-1")
    assert d.seen("msg-1") is True


def test_different_ids_are_independent():
    d = MessageDedup()
    assert d.seen("a") is False
    assert d.seen("b") is False
    assert d.seen("a") is True
    assert d.seen("b") is True


def test_empty_id_never_marked_as_seen():
    d = MessageDedup()
    assert d.seen("") is False
    assert d.seen("") is False


def test_lru_eviction_when_capacity_exceeded():
    d = MessageDedup(max_size=3)
    d.seen("a")
    d.seen("b")
    d.seen("c")
    d.seen("d")  # expulsa "a", cache agora = {b, c, d}
    assert d.seen("b") is True
    assert d.seen("c") is True
    assert d.seen("d") is True
    # "a" foi expulso — volta como novo (retorna False).
    assert d.seen("a") is False


def test_thread_safety_under_contention():
    d = MessageDedup()
    seen_results = []

    def worker():
        seen_results.append(d.seen("shared-id"))

    threads = [Thread(target=worker) for _ in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Exatamente uma thread deve ter visto "False" (primeira) e as demais "True".
    assert seen_results.count(False) == 1
    assert seen_results.count(True) == 19
