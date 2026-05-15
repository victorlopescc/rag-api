"""Testa o dependency de autenticação por API key."""
import pytest
from fastapi import HTTPException

from auth import require_api_key


def test_valid_key_passes():
    # Não deve levantar.
    require_api_key("test-secret")


def test_invalid_key_raises_401():
    with pytest.raises(HTTPException) as exc:
        require_api_key("wrong")
    assert exc.value.status_code == 401


def test_missing_key_raises_401():
    with pytest.raises(HTTPException) as exc:
        require_api_key(None)
    assert exc.value.status_code == 401
