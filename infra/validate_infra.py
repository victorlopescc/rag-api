#!/usr/bin/env python3
"""
Roda este script após subir a infra para validar que tudo está ok.
Uso: python validate_infra.py
"""

import sys
import subprocess
import urllib.request
import urllib.error
import json

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

def check_ollama():
    req = urllib.request.Request("http://localhost:11434/api/tags")
    with urllib.request.urlopen(req, timeout=5) as res:
        data = json.loads(res.read())
    models = [m["name"] for m in data.get("models", [])]
    if not models:
        raise RuntimeError("Nenhum modelo baixado ainda")
    return f"modelos: {', '.join(models)}"

def check_mistral():
    req = urllib.request.Request(
        "http://localhost:11434/api/tags"
    )
    with urllib.request.urlopen(req, timeout=5) as res:
        data = json.loads(res.read())
    models = [m["name"] for m in data.get("models", [])]
    has_llm = any("mistral" in m or "llama" in m for m in models)
    has_embed = any("nomic" in m for m in models)
    if not has_llm:
        raise RuntimeError("Modelo LLM não encontrado — rode: ollama pull mistral")
    if not has_embed:
        raise RuntimeError("Embedding não encontrado — rode: ollama pull nomic-embed-text")
    return "LLM + embedding prontos"

def check_ollama_generate():
    payload = json.dumps({
        "model": "mistral",
        "prompt": "Responda apenas: ok",
        "stream": False
    }).encode()
    req = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as res:
        data = json.loads(res.read())
    resp = data.get("response", "").strip()
    return f'resposta: "{resp[:30]}"'

print("\n=== Validação da infraestrutura RAG ===\n")

results = [
    check("Docker",                   check_docker),
    check("PostgreSQL (container)",   check_postgres),
    check("Ollama API",               check_ollama),
    check("Modelos instalados",       check_mistral),
    check("Geração de texto (lento)", check_ollama_generate),
]

print()
total = len(results)
passed = sum(results)
if passed == total:
    print(f"  Tudo certo! {passed}/{total} verificações passaram.\n")
else:
    print(f"  {passed}/{total} verificações passaram. Corrija os erros acima.\n")
    sys.exit(1)