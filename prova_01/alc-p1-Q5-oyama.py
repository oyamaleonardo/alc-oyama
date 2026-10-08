import numpy as np


def resolve_lu(A, b):
    n = len(A)
    L = np.eye(n)                # L comeca como identidade (diagonal igual a 1)
    U = np.array(A, dtype=float)  # U comeca como copia de A

    # Decomposicao A = LU (eliminacao de Gauss sem pivoteamento)
    for k in range(n):
        if U[k, k] == 0:
            raise Exception(
                "Pivo nulo encontrado: a decomposicao LU sem pivoteamento "
                "nao pode continuar. Utilize uma funcao alternativa, com "
                "pivoteamento, para resolver o sistema."
            )
        for i in range(k + 1, n):
            m = U[i, k] / U[k, k]  # multiplicador da linha i na etapa k
            L[i, k] = m            # o multiplicador e o elemento de L
            for j in range(k, n):
                U[i, j] = U[i, j] - m * U[k, j]

    # Substituicao progressiva: L y = b
    y = np.zeros(n)
    for i in range(n):
        soma = b[i]
        for j in range(i):
            soma = soma - L[i, j] * y[j]
        y[i] = soma  # a diagonal de L vale 1, entao nao ha divisao

    # Substituicao regressiva: U x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        soma = y[i]
        for j in range(i + 1, n):
            soma = soma - U[i, j] * x[j]
        x[i] = soma / U[i, i]

    return L, U, x


if __name__ == "__main__":
    A = np.array([[2.0, 1.0, 1.0],
                  [4.0, 3.0, 3.0],
                  [8.0, 7.0, 9.0]])
    b = np.array([4.0, 10.0, 24.0])

    L, U, x = resolve_lu(A, b)
    print("L =")
    print(L)
    print("U =")
    print(U)
    print("x =", x)