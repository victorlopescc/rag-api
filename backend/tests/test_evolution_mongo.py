"""Testa services.evolution_mongo com MongoClient mockado."""
from unittest.mock import MagicMock, patch

import services.evolution_mongo as mongo_module
from services.evolution_mongo import find_message_ids_for_lid


def _fake_db(docs):
    db = MagicMock()
    db.messageUpdate.find.return_value = iter(docs)
    return db


def test_returns_ids_from_mongo_cursor():
    fake = _fake_db([{"id": "msg-1"}, {"id": "msg-2"}])
    mongo_module._db.cache_clear()

    with patch.object(mongo_module, "_db", return_value=fake):
        ids = find_message_ids_for_lid("240247:6@lid")

    assert ids == ["msg-1", "msg-2"]
    query = fake.messageUpdate.find.call_args.args[0]
    assert query["fromMe"] is True
    assert "240247" in query["remoteJid"]["$regex"]


def test_skips_docs_without_id_field():
    fake = _fake_db([{"id": "ok"}, {}, {"id": None}])
    mongo_module._db.cache_clear()

    with patch.object(mongo_module, "_db", return_value=fake):
        ids = find_message_ids_for_lid("240247@lid")

    assert ids == ["ok"]


def test_empty_local_returns_empty():
    mongo_module._db.cache_clear()
    assert find_message_ids_for_lid("@lid") == []


def test_returns_empty_on_mongo_exception():
    mongo_module._db.cache_clear()

    def boom():
        raise RuntimeError("mongo offline")

    with patch.object(mongo_module, "_db", side_effect=boom):
        assert find_message_ids_for_lid("240247@lid") == []
