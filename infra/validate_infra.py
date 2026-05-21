#!/usr/bin/env python3
"""
Roda este script após subir a infra para validar que tudo está ok.
Uso: python validate_infra.py
"""

import os
import sys
import subprocess
import urllib.request
import urllib.error
import json
from pathlib import Path

OK  = "\033[92m✓\033[0m"
ERR = "\033[91m✗\033[0m"
WARN = "\033[93m⚠\033[0m"


def check(label, fn):
    try:
        result = fn()
        print(f"  {OK}  {label}" + (f" — {result}" if result else ""))
        return True
    except Exception as e:
        print(f"  {ERR}  {label} — {e}")
        return False


def check_docker():
    r = subprocess.run(["docker", "info"], capture_output=True, timeout=5)
    if r.returncode != 0:
        raise RuntimeError("Docker não está rodando")
    return "rodando"


def check_postgres():
    r = subprocess.run(
        ["docker", "exec", "rag_postgres", "pg_isready", "-U", "raguser", "-d", "ragdb"],
        capture_output=True, timeout=5
    )
    if r.returncode != 0:
        raise RuntimeError("PostgreSQL não responde")
    return "aceitando conexões"


def _load_env() -> dict:
    """Lê o .env irmão pra pegar a GEMINI_API_KEY. Não usamos pydantic
    aqui pra manter o script standalone (sem depender do venv do app)."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return {}
    out = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def check_gemini():
    env = _load_env()
    key = env.get("GEMINI_API_KEY") or os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY não definida no .env")
    # Ping leve: lista modelos do Gemini (não consome quota de geração).
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models?key={key}",
    )
    try:
        with urllib.request.urlopen(req, timeout=10) as res:
            data = json.loads(res.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:200]
        raise RuntimeError(f"HTTP {e.code} — {body}")
    n = len(data.get("models", []))
    if n == 0:
        raise RuntimeError("API respondeu mas sem modelos — key inválida?")
    return f"{n} modelos disponíveis"


print("\n=== Validação da infraestrutura RAG ===\n")

results = [
    check("Docker",                 check_docker),
    check("PostgreSQL (container)", check_postgres),
    check("Gemini API",             check_gemini),
]

print()
total = len(results)
passed = sum(results)
if passed == total:
    print(f"  Tudo certo! {passed}/{total} verificações passaram.\n")
else:
    print(f"  {passed}/{total} verificações passaram. Corrija os erros acima.\n")
    sys.exit(1)
