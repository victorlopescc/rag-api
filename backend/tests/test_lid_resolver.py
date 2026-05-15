"""Testa services.lid_resolver com Mongo e DB mockados."""
from unittest.mock import MagicMock, patch

from services.lid_resolver import resolve_student_by_lid


def _db_with_first_lid_hit(student):
    """DB que retorna `student` na primeira consulta (match por lid)."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.return_value = student
    return db


def _db_with_lookup_sequence(first_result, second_result):
    """DB cujo .first() retorna `first_result`, depois `second_result`."""
    db = MagicMock()
    db.query.return_value.filter.return_value.first.side_effect = [
        first_result,
        second_result,
    ]
    return db


def test_returns_student_matched_by_lid_directly():
    student = MagicMock(lid="240247@lid")
    db = _db_with_first_lid_hit(student)

    result = resolve_student_by_lid("240247:6@lid", db)

    assert result is student
    db.commit.assert_not_called()


def test_returns_none_when_no_lid_match_and_no_acks():
    db = _db_with_first_lid_hit(None)

    with patch("services.lid_resolver.find_message_ids_for_lid", return_value=[]):
        result = resolve_student_by_lid("240247@lid", db)

    assert result is None


def test_binds_lid_via_mongo_ack_correlation():
    pending_student = MagicMock(
        spec=["lid", "pending_welcome_id", "phone_number"],
        phone_number="5511999999999",
    )
    db = _db_with_lookup_sequence(None, pending_student)

    with patch(
        "services.lid_resolver.find_message_ids_for_lid",
        return_value=["msg-id-1", "msg-id-2"],
    ):
        result = resolve_student_by_lid("240247:6@lid", db)

    assert result is pending_student
    assert pending_student.lid == "240247@lid"  # normalizado e gravado
    assert pending_student.pending_welcome_id is None
    db.commit.assert_called_once()


def test_returns_none_when_acks_match_no_student():
    db = _db_with_lookup_sequence(None, None)

    with patch(
        "services.lid_resolver.find_message_ids_for_lid",
        return_value=["unknown-id"],
    ):
        result = resolve_student_by_lid("240247@lid", db)

    assert result is None
    db.commit.assert_not_called()
