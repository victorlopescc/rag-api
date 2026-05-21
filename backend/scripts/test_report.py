"""Roda todas as perguntas do roteiro de testes e gera um relatório.

Uso:
    cd backend && python -m scripts.test_report > relatorio.txt

Cobre Partes 1 e 2 (velocidade + qualidade RAG). Partes 3-7
(edge cases, poll, escalação, thread) exigem fluxo do webhook
com estado e devem ser validadas manualmente via WhatsApp.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass

# Encoding seguro pro console Windows.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

# Suprime os warnings barulhentos de cold-start.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")


@dataclass
class Case:
    tag: str
    question: str
    expected: list[str]      # qualquer substring → match (case-insensitive)
    category: str
    note: str = ""           # info extra (ex: "verifique se vem completo")


CASES: list[Case] = [
    # ============================================================
    # PARTE 2A — Grade tabular (15)
    # ============================================================
    Case("Q6",  "Qual a carga horária de AEDs1?",                    ["120"],            "Grade"),
    Case("Q7",  "Qual a carga horária de Cálculo I?",                ["80"],             "Grade"),
    Case("Q8",  "Quantas horas tem TGC?",                            ["120"],            "Grade"),
    Case("Q9",  "Qual o pré-requisito de AEDs2?",                    ["AEDs1", "Estruturas de Dados I"], "Grade"),
    Case("Q10", "Qual o pré-requisito de Arquitetura de Computadores I?", ["AEDs1", "Estruturas de Dados I"], "Grade"),
    Case("Q11", "Qual o pré-requisito de Engenharia de Software I?", ["AEDs2", "Estruturas de Dados II"], "Grade"),
    Case("Q12", "Qual o pré-requisito de TCC2?",                     ["TCC1", "TCC I", "TCC 1"], "Grade"),
    Case("Q13", "Em que período está AEDs1?",                        ["1"],              "Grade"),
    Case("Q14", "Em que período está TGC?",                          ["4"],              "Grade"),
    Case("Q15", "Quais disciplinas tem no 1º período?",              ["AEDs1", "Algoritmos"], "Grade", "lista parcial pode ser OK"),
    Case("Q16", "Quais disciplinas tem no 2º período?",              ["AEDs2", "Estruturas de Dados II"], "Grade", "lista parcial pode ser OK"),
    Case("Q17", "O que significa a sigla AEDs1?",                    ["Algoritmos", "Estruturas de Dados"], "Grade"),
    Case("Q18", "O que significa a sigla TGC?",                      ["Teoria dos Grafos", "Computabilidade"], "Grade"),
    Case("Q19", "O que é DIW?",                                      ["Desenvolvimento de Interfaces Web"], "Grade"),
    Case("Q20", "AEDs1 é pré-requisito de quais disciplinas?",       ["AEDs2", "AC1", "Arquitetura"], "Grade"),

    # ============================================================
    # PARTE 2B — TCC (8)
    # ============================================================
    Case("Q21", "Qual a duração mínima do TCC?",                     ["semestre", "2", "dois", "TCC1", "TCC2"], "TCC"),
    Case("Q22", "Quem orienta o TCC?",                               ["orientador", "professor"], "TCC"),
    Case("Q23", "Quem é o coordenador de TCC?",                      [],                 "TCC", "info pode não estar no doc"),
    Case("Q24", "Posso fazer TCC em dupla?",                         ["dupla", "individual", "grupo"], "TCC"),
    Case("Q25", "Qual o formato da defesa?",                         ["banca", "apresent", "oral"], "TCC"),
    Case("Q26", "Como faço para mudar de orientador?",               ["orientador"],     "TCC", "info pode não estar no doc"),
    Case("Q27", "O TCC pode ser sobre qualquer tema?",               ["tema", "comput", "ciênc"], "TCC"),
    Case("Q28", "Qual a nota mínima de aprovação?",                  ["6", "7", "70", "60"], "TCC", "verifique se faz sentido"),

    # ============================================================
    # PARTE 2C — ADA (8)
    # ============================================================
    Case("Q29", "Quando vai ser a ADA?",                             ["data", "junho", "/06", "horário", "Dia"], "ADA"),
    Case("Q30", "Quanto vale a prova da ADA?",                       ["5", "cinco", "ponto"], "ADA"),
    Case("Q31", "Posso usar calculadora na ADA?",                    ["calculadora"],    "ADA"),
    Case("Q32", "Quantas questões tem a ADA?",                       [],                 "ADA", "verifique manualmente"),
    Case("Q33", "Qual a duração da ADA?",                            [],                 "ADA", "info pode não estar no doc"),
    Case("Q34", "O que acontece se eu não fizer a ADA?",             ["just", "Guia do Aluno", "ausência", "compare"], "ADA"),
    Case("Q35", "A ADA é obrigatória?",                              ["obrigatóri", "sim"], "ADA"),
    Case("Q36", "Tem ADA substitutiva?",                             ["substitut", "just"], "ADA", "pode dar fallback se não tiver no doc"),

    # ============================================================
    # PARTE 2D — PPC / Estrutura (6)
    # ============================================================
    Case("Q37", "Qual a duração total do curso de Ciência da Computação?", ["8", "oito", "4", "quatro", "anos", "períodos", "semestres"], "PPC"),
    Case("Q38", "Qual a carga horária total obrigatória?",           ["3.520", "3520", "3.200", "3200"], "PPC"),
    Case("Q39", "Qual a carga horária total das optativas?",         ["320"],            "PPC"),
    Case("Q40", "Quantas disciplinas tem no curso?",                 ["56", "55", "57", "58"], "PPC", "exige contagem"),
    Case("Q41", "O curso é presencial ou online?",                   ["presencial", "online", "híbrid", "EaD"], "PPC"),
    Case("Q42", "Sou aluno do 5º período, vou fazer estágio obrigatório?", ["estágio", "obrigatório", "opcional"], "PPC"),
]


# ============================================================
# Velocidade (Parte 1)
# ============================================================
VELOCITY_QUERIES = [
    "Quando vai ser a ADA?",
    "Qual a duração mínima do TCC?",
    "Posso usar calculadora?",
    "Qual a carga horária de AEDs1?",
    "Quais disciplinas tem no 5º período?",
]


def check(answer: str, expected: list[str]) -> bool:
    """OK se qualquer expected substring está na answer (case-insensitive).
    Se expected vazio, returns False (caso "ambíguo, verificar manualmente").
    """
    if not expected:
        return False
    low = (answer or "").lower()
    return any(e.lower() in low for e in expected)


def main() -> int:
    # Setup uma vez
    print("=" * 72)
    print("INICIALIZANDO PIPELINE...")
    print("=" * 72)
    from pipeline import bm25_index
    bm25_index.build()
    from rag_engine import ask
    print("BM25 ok. Aquecendo Gemini com 1 ping...")
    _ = ask("teste")
    print("Pronto.\n")

    # -------------------- PARTE 1: VELOCIDADE --------------------
    print("=" * 72)
    print("PARTE 1 — VELOCIDADE (cronômetro)")
    print("=" * 72)
    velocities = []
    for i, q in enumerate(VELOCITY_QUERIES, 1):
        t0 = time.time()
        r = ask(q)
        dt = time.time() - t0
        velocities.append(dt)
        flag = "OK" if dt < 8 else "LENTO"
        print(f"  Q{i} {dt:5.1f}s  [{flag}]  {q}")
    avg = sum(velocities) / len(velocities)
    print(f"\n  Latência média: {avg:.1f}s  (meta: < 5s)\n")

    # -------------------- PARTE 2: QUALIDADE --------------------
    print("=" * 72)
    print("PARTE 2 — QUALIDADE DAS RESPOSTAS")
    print("=" * 72)

    results: list[tuple[Case, str, float, bool, bool]] = []
    # (case, answer, latency, ok, was_fallback)

    last_cat = None
    for case in CASES:
        if case.category != last_cat:
            print(f"\n--- {case.category} ---")
            last_cat = case.category

        t0 = time.time()
        r = ask(case.question)
        dt = time.time() - t0
        ok = check(r.answer, case.expected) if not r.was_fallback else False
        fb = r.was_fallback

        if ok:
            mark = "[PASS]"
        elif fb and case.note and ("não estar" in case.note or "fallback se" in case.note):
            mark = "[FB-OK]"  # fallback esperado
        elif fb:
            mark = "[FB]"
        else:
            mark = "[FAIL]"

        results.append((case, r.answer, dt, ok, fb))
        print(f"  {mark:8s} {dt:5.1f}s {case.tag:4s} {case.question}")
        print(f"           → {r.answer[:140]}")
        if case.note:
            print(f"           (nota: {case.note})")

    # -------------------- RESUMO --------------------
    print("\n" + "=" * 72)
    print("RESUMO")
    print("=" * 72)

    # Por categoria
    by_cat: dict[str, list] = {}
    for case, ans, dt, ok, fb in results:
        by_cat.setdefault(case.category, []).append((case, ok, fb))

    for cat, items in by_cat.items():
        passed = sum(1 for _, ok, _ in items if ok)
        fb_expected = sum(
            1 for c, ok, fb in items
            if fb and c.note and ("não estar" in c.note or "fallback se" in c.note)
        )
        total = len(items)
        effective = passed + fb_expected  # fallbacks esperados contam
        pct = effective / total * 100 if total else 0
        print(f"  {cat:10s} {passed}/{total} pass  |  +{fb_expected} fallbacks legítimos  |  {effective}/{total} efetivo ({pct:.0f}%)")

    total_pass = sum(1 for _, _, _, ok, _ in results if ok)
    total_fb_ok = sum(
        1 for c, _, _, _, fb in results
        if fb and c.note and ("não estar" in c.note or "fallback se" in c.note)
    )
    total = len(results)
    print(f"\n  GERAL     {total_pass}/{total} pass  |  +{total_fb_ok} fallbacks legítimos  |  {(total_pass + total_fb_ok)}/{total} efetivo ({(total_pass + total_fb_ok)/total*100:.0f}%)")

    # Latência
    lats = [dt for _, _, dt, _, _ in results]
    print(f"\n  Latência RAG: avg={sum(lats)/len(lats):.1f}s  min={min(lats):.1f}s  max={max(lats):.1f}s")

    # Falhas a investigar
    print("\n" + "-" * 72)
    print("CASOS A REVISAR MANUALMENTE:")
    print("-" * 72)
    for case, ans, dt, ok, fb in results:
        if not ok:
            note = f" — {case.note}" if case.note else ""
            tag = "FB" if fb else "FAIL"
            print(f"  [{tag}] {case.tag} {case.question}{note}")
            print(f"        → {ans[:220]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
