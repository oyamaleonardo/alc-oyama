# -*- coding: utf-8 -*-
"""
Álgebra Linear Computacional - Lista de Exercícios I
4a Questão

Implementação do algoritmo de substituição regressiva (back substitution) para
resolver o sistema linear U x = b, onde:
    a) U é uma matriz triangular superior (n x n); e
    b) b é um vetor coluna de tamanho correspondente (n x 1).

Conforme solicitado no enunciado, a implementação levanta uma exception
(DiagonalNulaError) caso seja identificado um elemento nulo na diagonal de U.

Bibliotecas usadas: NumPy e SciPy.
"""

import sys

import numpy as np
from scipy.linalg import solve_triangular  # usado apenas na verificação do main


class DiagonalNulaError(ValueError):
    """Elemento nulo na diagonal principal da matriz triangular superior.

    Nesse caso a matriz é singular e o sistema U x = b não possui solução única,
    de modo que a substituição regressiva não pode ser executada.
    """


def substituicao_regressiva(U, b, tol=1e-12):
    """Resolve U x = b por substituição regressiva.

    Parâmetros
    ----------
    U : array_like, shape (n, n)
        Matriz triangular superior.
    b : array_like, shape (n,) ou (n, 1)
        Vetor coluna de termos independentes.
    tol : float, opcional
        Tolerância para considerar um elemento da diagonal como nulo
        (|u_ii| <= tol). Use tol=0.0 para exigir zero exato.

    Retorna
    -------
    x : ndarray
        Solução do sistema, com o mesmo formato do vetor b fornecido.

    Levanta
    -------
    DiagonalNulaError
        Se algum elemento da diagonal de U for nulo (|u_ii| <= tol).
    ValueError
        Se U não for quadrada, se b não for um vetor coluna compatível
        ou se U não for triangular superior.
    """
    U = np.asarray(U, dtype=float)
    b = np.asarray(b, dtype=float)

    # --- Validação da matriz -------------------------------------------------
    if U.ndim != 2:
        raise ValueError(f"U deve ser uma matriz 2-D; recebido ndim={U.ndim}.")

    n, m = U.shape
    if n != m:
        raise ValueError(f"U deve ser quadrada; recebido shape {U.shape}.")
    if n == 0:
        raise ValueError("U deve ter ao menos uma linha e uma coluna.")

    # Entrada declarada como triangular superior: parte estritamente inferior nula.
    if np.any(np.abs(np.tril(U, k=-1)) > tol):
        raise ValueError("U não é triangular superior (há elementos não nulos abaixo da diagonal).")

    # --- Validação do vetor coluna ------------------------------------------
    formato_coluna = (b.ndim == 2)
    if b.ndim == 1:
        b_vet = b
    elif b.ndim == 2 and b.shape[1] == 1:
        b_vet = b[:, 0]
    else:
        raise ValueError(f"b deve ser um vetor coluna (n, 1) ou (n,); recebido shape {b.shape}.")

    if b_vet.shape[0] != n:
        raise ValueError(f"Dimensões incompatíveis: U é {n}x{n} e b tem {b_vet.shape[0]} elementos.")

    # --- Verificação da diagonal --------------------------------------------
    diagonal = np.diag(U)
    nulos = np.flatnonzero(np.abs(diagonal) <= tol)
    if nulos.size > 0:
        i = int(nulos[0])
        raise DiagonalNulaError(
            f"Elemento nulo na diagonal: U[{i}][{i}] = {diagonal[i]:.3e} "
            f"(|u_ii| <= tol = {tol:.1e}). A matriz é singular."
        )

    # --- Substituição regressiva --------------------------------------------
    # x_i = ( b_i - sum_{j=i+1}^{n-1} u_ij * x_j ) / u_ii , para i = n-1, ..., 0
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        soma = np.dot(U[i, i + 1:], x[i + 1:])
        x[i] = (b_vet[i] - soma) / U[i, i]

    return x.reshape(n, 1) if formato_coluna else x


# ---------------------------------------------------------------------------
# Demonstração / testes
# ---------------------------------------------------------------------------
def _mostrar(titulo):
    print()
    print("=" * 70)
    print(titulo)
    print("=" * 70)


def main():
    np.set_printoptions(precision=6, suppress=True)

    # ---- Caso 1: sistema 3x3 com solução conhecida ------------------------
    _mostrar("Caso 1 - Sistema 3x3")
    U1 = np.array([[2.0, -1.0,  3.0],
                   [0.0,  4.0, -2.0],
                   [0.0,  0.0,  5.0]])
    b1 = np.array([[5.0],
                   [6.0],
                   [10.0]])
    x1 = substituicao_regressiva(U1, b1)
    print("U =\n", U1)
    print("b =\n", b1)
    print("x =\n", x1)
    print("Resíduo ||U x - b||_2 =", np.linalg.norm(U1 @ x1 - b1))

    # ---- Caso 2: comparação com SciPy em matriz aleatória -----------------
    _mostrar("Caso 2 - Matriz aleatória 6x6 (comparação com scipy.linalg.solve_triangular)")
    rng = np.random.default_rng(42)
    U2 = np.triu(rng.uniform(-5.0, 5.0, size=(6, 6)))
    U2[np.diag_indices(6)] += 6.0          # afasta a diagonal de zero
    b2 = rng.uniform(-10.0, 10.0, size=(6, 1))
    x2 = substituicao_regressiva(U2, b2)
    x_ref = solve_triangular(U2, b2, lower=False)
    print("x  (implementação) =", x2.ravel())
    print("x  (SciPy)         =", x_ref.ravel())
    print("Diferença máxima   =", np.max(np.abs(x2 - x_ref)))
    print("Resíduo ||U x - b||_2 =", np.linalg.norm(U2 @ x2 - b2))

    # ---- Caso 3: aceita vetor 1-D e devolve 1-D ---------------------------
    _mostrar("Caso 3 - Entrada como vetor 1-D")
    x3 = substituicao_regressiva(U1, np.array([5.0, 6.0, 10.0]))
    print("x =", x3, " (shape", x3.shape, ")")

    # ---- Caso 4: elemento nulo na diagonal -> exception --------------------
    _mostrar("Caso 4 - Elemento nulo na diagonal (deve levantar exception)")
    U4 = np.array([[1.0, 2.0, 3.0],
                   [0.0, 0.0, 4.0],
                   [0.0, 0.0, 5.0]])
    b4 = np.array([[1.0], [2.0], [3.0]])
    try:
        substituicao_regressiva(U4, b4)
    except DiagonalNulaError as erro:
        print("DiagonalNulaError levantada com sucesso:")
        print("  ", erro)
    else:
        print("ERRO: a exception esperada não foi levantada.")

    # ---- Caso 5: dimensões incompatíveis ----------------------------------
    _mostrar("Caso 5 - Dimensões incompatíveis (deve levantar exception)")
    try:
        substituicao_regressiva(U1, np.array([[1.0], [2.0]]))
    except ValueError as erro:
        print("ValueError levantada com sucesso:")
        print("  ", erro)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    main()
