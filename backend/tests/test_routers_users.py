"""Testa o router /users com Evolution client mockado."""
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from database import Student, get_db
from main import app


@pytest.fixture
def client(mock_db):
    # SQLAlchemy só aplica defaults na flush — como o mock não flusha, precisamos
    # popular `id` e `active` manualmente no refresh para o _to_out funcionar.
    def refresh(obj):
        if getattr(obj, "id", None) is None:
            obj.id = uuid.uuid4()
        if getattr(obj, "active", None) is None:
            obj.active = True
        if getattr(obj, "data_consent", None) is None:
            obj.data_consent = True
    mock_db.refresh.side_effect = refresh

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _student(phone="5511999999999"):
    s = Student(
        id=uuid.uuid4(),
        full_name="Maria Silva",
        matricula="20250001",
        phone_number=phone,
        lid=None,
        active=True,
    )
    return s


def test_register_creates_student_and_sends_welcome(client, mock_db):
    # Primeiro filter retorna None (não existe); depois o add insere.
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch(
        "routers.users.evolution_client.send_text",
        new_callable=AsyncMock,
        return_value="welcome-msg-id",
    ):
        resp = client.post("/users/register", json={
            "full_name": "Maria Silva",
            "matricula": "20250001",
            "phone_number": "11999999999",
        })

    assert resp.status_code == 200
    body = resp.json()
    assert body["phone_number"] == "5511999999999"  # normalizado com DDI 55
    assert body["full_name"] == "Maria Silva"


def test_register_updates_existing_student(client, mock_db):
    existing = _student()
    mock_db.query.return_value.filter.return_value.first.return_value = existing

    with patch(
        "routers.users.evolution_client.send_text",
        new_callable=AsyncMock,
        return_value="welcome-msg-id",
    ):
        resp = client.post("/users/register", json={
            "full_name": "Maria Nova",
            "matricula": "20250099",
            "phone_number": "11999999999",
        })

    assert resp.status_code == 200
    assert existing.full_name == "Maria Nova"
    assert existing.matricula == "20250099"


def test_register_succeeds_even_if_whatsapp_fails(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None

    with patch(
        "routers.users.evolution_client.send_text",
        new_callable=AsyncMock,
        side_effect=Exception("Evolution offline"),
    ):
        resp = client.post("/users/register", json={
            "full_name": "Ana",
            "matricula": "001",
            "phone_number": "11988888888",
        })

    assert resp.status_code == 200


def test_list_students_returns_all(client, mock_db):
    mock_db.query.return_value.order_by.return_value.all.return_value = [_student(), _student("5511988888888")]
    resp = client.get("/users")
    assert resp.status_code == 200
    assert len(resp.json()) == 2


def test_delete_student_requires_api_key(client):
    resp = client.delete("/users/5511999999999")
    assert resp.status_code == 401


def test_delete_student_not_found(client, api_headers, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    resp = client.delete("/users/5511999999999", headers=api_headers)
    assert resp.status_code == 404


def test_delete_student_success(client, api_headers, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = _student()
    resp = client.delete("/users/5511999999999", headers=api_headers)
    assert resp.status_code == 200
    mock_db.delete.assert_called_once()


def test_register_without_consent_is_rejected(client, mock_db):
    """Sem consentimento o cadastro deve falhar (400)."""
    resp = client.post("/users/register", json={
        "full_name": "Ana",
        "matricula": "001",
        "phone_number": "11988888888",
        "data_consent": False,
    })
    assert resp.status_code == 400
    assert "dados" in resp.json()["detail"].lower()


def test_register_with_explicit_consent_true(client, mock_db):
    mock_db.query.return_value.filter.return_value.first.return_value = None
    with patch(
        "routers.users.evolution_client.send_text",
        new_callable=AsyncMock, return_value="id",
    ):
        resp = client.post("/users/register", json={
            "full_name": "Bea",
            "matricula": "002",
            "phone_number": "11977777777",
            "data_consent": True,
        })
    assert resp.status_code == 200
    assert resp.json()["data_consent"] is True
