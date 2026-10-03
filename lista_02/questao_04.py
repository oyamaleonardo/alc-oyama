# -*- coding: utf-8 -*-
"""
Álgebra Linear Computacional - Lista de Exercícios II
4a Questão - [FORD] Exercício 7.32 (em Python ao invés do MATLAB)

O produto interno de dois vetores u e v (n x 1) é o número real <u, v> = u^T v.
O exercício investiga a matriz n x n A = u v^T (produto externo). Para
n = 5, 15, 25, gera-se u = rand(n, 1) e v = rand(n, 1) e calcula-se:
    a) posto(u v^T);
    b) ||u||_2 ||v||_2; e
    c) ||u v^T||_2.

Conclusões do experimento (demonstradas na solução escrita), para u, v != 0:
    Afirmação 1) posto(u v^T) = 1;
    Afirmação 2) ||u v^T||_2 = ||u||_2 ||v||_2.

Cada bloco da demonstração em main() verifica numericamente um passo das
demonstrações. Os títulos impressos indicam a afirmação e o passo.

Bibliotecas usadas: NumPy e SciPy.
"""

import sys

import numpy as np
from scipy.linalg import null_space


def produto_externo(u, v):
    """Retorna a matriz n x n A = u v^T a partir de dois vetores de tamanho n.

    Aceita vetores 1-D (n,) ou coluna (n, 1).
    """
    u = np.asarray(u, dtype=float).reshape(-1, 1)
    v = np.asarray(v, dtype=float).reshape(-1, 1)
    if u.shape != v.shape:
        raise ValueError(f"u e v devem ter o mesmo tamanho; recebido {u.shape[0]} e {v.shape[0]}.")
    return u @ v.T


def experimento(n, rng):
    """Executa o experimento do enunciado para um valor de n.

    Retorna um dicionário com os vetores, a matriz e as três grandezas pedidas.
    """
    # Equivalente a rand(n,1) do MATLAB: distribuição uniforme em [0, 1).
    u = rng.random((n, 1))
    v = rng.random((n, 1))
    A = produto_externo(u, v)

    return {
        "n": n,
        "u": u,
        "v": v,
        "A": A,
        "posto": np.linalg.matrix_rank(A),
        "produto_normas": np.linalg.norm(u.ravel()) * np.linalg.norm(v.ravel()),
        "norma_A": np.linalg.norm(A, 2),  # norma 2 = maior valor singular de A
    }


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------
def _mostrar(titulo):
    print()
    print("=" * 78)
    print(titulo)
    print("=" * 78)


def main(semente=None):
    rng = np.random.default_rng(semente)
    eps = np.finfo(float).eps

    resultados = [experimento(n, rng) for n in (5, 15, 25)]

    # ---- Experimento do enunciado ------------------------------------------
    _mostrar("Experimento do enunciado: A = u v^T, com u = rand(n,1) e v = rand(n,1)")
    print(f"{'n':>4} | {'posto(uv^T)':>11} | {'||u||_2 ||v||_2':>18} | "
          f"{'||uv^T||_2':>18} | {'diferença':>10}")
    print("-" * 78)
    for r in resultados:
        dif = abs(r["norma_A"] - r["produto_normas"])
        print(f"{r['n']:>4} | {r['posto']:>11d} | {r['produto_normas']:>18.12f} | "
              f"{r['norma_A']:>18.12f} | {dif:>10.2e}")

    # ---- Afirmação 1, passos 2 e 3: N(A) = {x : v^T x = 0}, dim = n - 1 ----
    # O SciPy calcula uma base do núcleo sem usar a demonstração. Conferimos
    # que todo vetor dessa base é ortogonal a v e que a base tem n - 1 vetores.
    _mostrar("Afirmação 1, passos 2 e 3: N(u v^T) = {x : v^T x = 0}, de dimensão n - 1")
    for r in resultados:
        N = null_space(r["A"])                      # base ortonormal do núcleo
        print(f"n = {r['n']:>2}:  dim N(A) = {N.shape[1]:>2}  (n - 1 = {r['n'] - 1:>2})   "
              f"max |v^T x| nos vetores da base = {np.max(np.abs(r['v'].T @ N)):.2e}")

    # ---- Afirmação 1, passo 4: posto = n - nulidade (Teorema 3.4) ----------
    _mostrar("Afirmação 1, passo 4: posto = n - nulidade (Teorema 3.4)")
    print(f"{'n':>4} | {'nulidade':>8} | {'n - nulidade':>12} | {'posto (matrix_rank)':>19}")
    print("-" * 54)
    for r in resultados:
        nulidade = null_space(r["A"]).shape[1]
        print(f"{r['n']:>4} | {nulidade:>8d} | {r['n'] - nulidade:>12d} | {r['posto']:>19d}")

    # ---- Afirmação 1, prova alternativa: Im(A) = span{u} -------------------
    # Cada coluna de A é projetada sobre u. Se ela pertence a span{u}, a parte
    # que sobra fora da direção de u (resíduo) é nula, e o coeficiente da
    # projeção da coluna j deve ser v_j.
    _mostrar("Afirmação 1, prova alternativa: Im(u v^T) = span{u} (coluna j = v_j u)")
    for r in resultados:
        u, A = r["u"], r["A"]
        coef = (u.T @ A) / (u.T @ u)                # coeficiente de cada coluna na direção de u
        residuo = A - u @ coef                      # parte de cada coluna fora de span{u}
        print(f"n = {r['n']:>2}:  max ||resíduo das colunas||_2 = "
              f"{np.max(np.linalg.norm(residuo, axis=0)):.2e}   "
              f"max |coef_j - v_j| = {np.max(np.abs(coef - r['v'].T)):.2e}")

    # ---- Afirmação 2, passos 1 e 2: ||A x||_2 <= ||u||_2 ||v||_2 ------------
    # Sorteamos 10 000 vetores unitários x: nenhum ultrapassa a cota de
    # Cauchy-Schwarz, e vetores sorteados ao acaso ficam abaixo dela.
    _mostrar("Afirmação 2, passos 1 e 2: ||A x||_2 <= ||u||_2 ||v||_2 para ||x||_2 = 1")
    for r in resultados:
        X = rng.standard_normal((r["n"], 10000))
        X /= np.linalg.norm(X, axis=0)              # 10 000 vetores unitários
        maior = np.max(np.linalg.norm(r["A"] @ X, axis=0))
        print(f"n = {r['n']:>2}:  max ||A x||_2 (10 000 x sorteados) = {maior:.6f}"
              f"   <=   ||u||_2 ||v||_2 = {r['produto_normas']:.6f}")

    # ---- Afirmação 2, passo 3: a cota é atingida em x = v / ||v||_2 --------
    _mostrar("Afirmação 2, passo 3: em x* = v / ||v||_2, ||A x*||_2 = ||u||_2 ||v||_2")
    for r in resultados:
        x_estrela = r["v"] / np.linalg.norm(r["v"].ravel())
        valor = np.linalg.norm(r["A"] @ x_estrela)
        print(f"n = {r['n']:>2}:  ||A x*||_2 = {valor:.12f}   ||u||_2 ||v||_2 = "
              f"{r['produto_normas']:.12f}   diferença = {abs(valor - r['produto_normas']):.2e}")

    # ---- Valores singulares e posto numérico -------------------------------
    # Em aritmética exata, sigma_1 = ||u|| ||v|| e sigma_2 = ... = sigma_n = 0.
    # Em ponto flutuante, sigma_2..sigma_n ficam da ordem de eps * sigma_1, e
    # matrix_rank só conta os valores singulares acima de tol = sigma_1 * n * eps.
    _mostrar("Valores singulares e posto numérico (tol = sigma_1 * n * eps, usada por matrix_rank)")
    for r in resultados:
        s = np.linalg.svd(r["A"], compute_uv=False)
        tol = s[0] * max(r["A"].shape) * eps
        print(f"n = {r['n']:>2}:  sigma_1 = {s[0]:.12f}   max(sigma_2..sigma_n) = {np.max(s[1:]):.2e}"
              f"   eps*sigma_1 = {eps * s[0]:.2e}   tol = {tol:.2e}")

    # ---- Caso n = 5 por extenso --------------------------------------------
    _mostrar("Caso n = 5 por extenso")
    r = resultados[0]
    np.set_printoptions(precision=4, suppress=True)
    print("u^T =", r["u"].ravel())
    print("v^T =", r["v"].ravel())
    print("A = u v^T =\n", r["A"])


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    # Passe um inteiro como argumento para resultados reprodutíveis:
    #   python questao_04.py 42
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
