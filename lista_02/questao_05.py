# -*- coding: utf-8 -*-
"""
Álgebra Linear Computacional - Lista de Exercícios II
5a Questão - [FORD] Exercícios 6.38 e 6.39 (em Python ao invés do MATLAB)

Uma matriz quadrada P é ortogonal quando P^T P = I, o que equivale a dizer
que suas colunas têm norma unitária e são ortogonais entre si.

As matrizes do enunciado têm apenas 5 algarismos significativos, por isso a
comparação usa uma tolerância (tol = 1e-4) em vez de igualdade exata.

Bibliotecas usadas: NumPy.
"""

import sys

import numpy as np


def is_orthogonal_by_definition(P, tol=1e-4):
    """Retorna True se P^T P = I (a menos de tol)."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    return bool(np.allclose(P.T @ P, np.eye(P.shape[0]), rtol=0.0, atol=tol))


def is_orthogonal_by_vectors(P, tol=1e-4):
    """Retorna True se as colunas de P têm norma 1 e são ortogonais duas a duas."""
    P = np.asarray(P, dtype=float)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        return False
    n = P.shape[1]
    for i in range(n):
        if abs(np.linalg.norm(P[:, i]) - 1.0) > tol:        # norma unitária
            return False
        for j in range(i + 1, n):
            if abs(np.dot(P[:, i], P[:, j])) > tol:          # ortogonalidade
                return False
    return True


# ---------------------------------------------------------------------------
# Matrizes do enunciado
# ---------------------------------------------------------------------------
MATRIZES = {
    "6.38 a)": np.array([[-0.40825,  0.43644,  0.80178],
                         [-0.8165,   0.21822, -0.53452],
                         [-0.40825, -0.87287,  0.26726]]),
    "6.38 b)": np.array([[-0.51450,  0.48507,  0.70711],
                         [-0.68599, -0.72761,  0.0],
                         [ 0.51450, -0.48507,  0.70711]]),
    "6.39 a)": np.array([[-0.58835,  0.70206,  0.40119],
                         [-0.78446, -0.37524, -0.49377],
                         [-0.19612, -0.60523,  0.77152]]),
    "6.39 b)": np.array([[-0.47624,  -0.4264,   0.30151],
                         [ 0.087932,  0.86603, -0.40825],
                         [-0.87491,  -0.26112,  0.86164]]),
}


def main():
    np.set_printoptions(precision=5, suppress=True)
    for nome, P in MATRIZES.items():
        print("=" * 60)
        print(f"Exercício {nome}")
        print("=" * 60)
        print("P^T P =\n", P.T @ P)
        print("Normas das colunas:", np.linalg.norm(P, axis=0))
        print("is_orthogonal_by_definition:", is_orthogonal_by_definition(P))
        print("is_orthogonal_by_vectors:   ", is_orthogonal_by_vectors(P))
        print()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    main()
