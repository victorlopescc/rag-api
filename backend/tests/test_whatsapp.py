"""Testa helpers puros do módulo services.whatsapp."""
import pytest

from services.whatsapp import (
    WELCOME_MESSAGE,
    build_welcome_text,
    normalize_lid,
    normalize_phone,
)


class TestNormalizeLid:
    def test_remove_device_suffix(self):
        assert normalize_lid("240247703105761:6@lid") == "240247703105761@lid"

    def test_without_device_suffix(self):
        assert normalize_lid("240247703105761@lid") == "240247703105761@lid"

    def test_empty_string_returns_empty(self):
        assert normalize_lid("") == ""

    def test_none_returns_none(self):
        assert normalize_lid(None) is None

    def test_non_lid_jid_is_returned_as_is(self):
        assert normalize_lid("5511999999999@s.whatsapp.net") == "5511999999999@s.whatsapp.net"


class TestNormalizePhone:
    def test_strips_formatting(self):
        assert normalize_phone("(11) 99999-9999") == "5511999999999"

    def test_adds_ddi_55_when_missing(self):
        assert normalize_phone("11999999999") == "5511999999999"

    def test_keeps_ddi_55_when_present(self):
        assert normalize_phone("5511999999999") == "5511999999999"

    def test_empty_returns_empty(self):
        assert normalize_phone("") == ""

    def test_none_returns_empty(self):
        assert normalize_phone(None) == ""

    def test_only_spaces(self):
        assert normalize_phone("   ") == ""


class TestBuildWelcomeText:
    def test_uses_first_name_only(self):
        txt = build_welcome_text("Maria da Silva")
        assert "Maria" in txt
        assert "Silva" not in txt

    def test_handles_single_name(self):
        assert "João" in build_welcome_text("João")

    def test_empty_name(self):
        # Não deve explodir — apenas produz um texto com nome vazio.
        txt = build_welcome_text("")
        assert isinstance(txt, str)

    def test_template_includes_welcome(self):
        assert "cadastro" in WELCOME_MESSAGE.lower()
