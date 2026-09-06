# -*- coding: utf-8 -*-
"""
Álgebra Linear Computacional - Lista de Exercícios I
5a Questão

Robô planar de dois elos e duas juntas, com L1 = 20,0 cm e L2 = 15,0 cm.

a) `posicao_efetuador(theta1, theta2)` recebe os ângulos das juntas e devolve a
   posição do efetuador final (X_U, Y_U) em cm, com precisão de 1 casa decimal.

b) `matriz_transformacao(theta1, theta2)` devolve a matriz 3x3 (coordenadas
   homogêneas) que leva coordenadas do referencial do efetuador final para o
   referencial (X_U, Y_U) fixado na primeira junta.

Convenções adotadas (conforme a Figura 1 do enunciado):
  - theta1 é medido a partir do eixo X_U, no sentido anti-horário;
  - theta2 é medido a partir do prolongamento do elo 1 (ângulo relativo), também
    no sentido anti-horário;
  - o referencial do efetuador (em vermelho na figura) tem origem na ponta do
    robô e seu eixo x alinhado com o elo 2.

Bibliotecas utilizadas: NumPy.
"""

import sys

import numpy as np

# Comprimentos dos elos, em centímetros.
L1 = 20.0
L2 = 15.0


# ---------------------------------------------------------------------------
# Matrizes elementares em coordenadas homogêneas (3x3)
# ---------------------------------------------------------------------------
def rotacao(theta):
    """Rotação de `theta` radianos em torno da origem, no sentido anti-horário."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0],
                     [s, c, 0.0],
                     [0.0, 0.0, 1.0]])


def translacao_x(comprimento):
    """Translação de `comprimento` ao longo do eixo x do referencial corrente."""
    return np.array([[1.0, 0.0, comprimento],
                     [0.0, 1.0, 0.0],
                     [0.0, 0.0, 1.0]])


def _para_radianos(theta1, theta2, graus):
    if graus:
        return np.radians(float(theta1)), np.radians(float(theta2))
    return float(theta1), float(theta2)


# ---------------------------------------------------------------------------
# Item (b) - matriz de transformação do efetuador para (X_U, Y_U)
# ---------------------------------------------------------------------------
def matriz_transformacao(theta1, theta2, l1=L1, l2=L2, graus=True):
    """Matriz 3x3 que leva do referencial do efetuador final para (X_U, Y_U).

    A transformação é a composição, nesta ordem, de:
        1) rotação de theta1                      -> alinha com o elo 1
        2) translação de L1 ao longo do novo x    -> chega à segunda junta
        3) rotação de theta2                      -> alinha com o elo 2
        4) translação de L2 ao longo do novo x    -> chega ao efetuador

            T = R(theta1) . D(L1) . R(theta2) . D(L2)

    Em forma fechada, com c12 = cos(theta1 + theta2) e s12 = sen(theta1 + theta2):

            [ c12  -s12   L1*cos(theta1) + L2*c12 ]
            [ s12   c12   L1*sen(theta1) + L2*s12 ]
            [  0     0              1             ]

    O bloco 2x2 superior esquerdo é a orientação do efetuador; a última coluna
    é a sua posição, ou seja, exatamente a resposta do item (a).

    Parâmetros
    ----------
    theta1, theta2 : float
        Ângulos das juntas (em graus, salvo se `graus=False`).
    l1, l2 : float, opcional
        Comprimentos dos elos, em cm.
    graus : bool, opcional
        True (padrão) interpreta os ângulos em graus; False, em radianos.

    Retorna
    -------
    T : ndarray, shape (3, 3)
    """
    t1, t2 = _para_radianos(theta1, theta2, graus)
    return rotacao(t1) @ translacao_x(l1) @ rotacao(t2) @ translacao_x(l2)


# ---------------------------------------------------------------------------
# Item (a) - posição do efetuador final
# ---------------------------------------------------------------------------
def posicao_efetuador(theta1, theta2, l1=L1, l2=L2, graus=True, arredondar=True):
    """Posição (X_U, Y_U) do efetuador final, em cm.

    Obtida da última coluna da matriz de transformação, o que equivale a:
        X_U = L1*cos(theta1) + L2*cos(theta1 + theta2)
        Y_U = L1*sen(theta1) + L2*sen(theta1 + theta2)

    Parâmetros
    ----------
    theta1, theta2 : float
        Ângulos das juntas (em graus, salvo se `graus=False`).
    l1, l2 : float, opcional
        Comprimentos dos elos, em cm.
    graus : bool, opcional
        True (padrão) interpreta os ângulos em graus; False, em radianos.
    arredondar : bool, opcional
        True (padrão) devolve os valores com 1 casa decimal, como pede o
        enunciado. False devolve a precisão total (útil em contas encadeadas).

    Retorna
    -------
    (x, y) : tuple de float
        Coordenadas do efetuador em cm.
    """
    T = matriz_transformacao(theta1, theta2, l1, l2, graus)
    x, y = float(T[0, 2]), float(T[1, 2])

    if arredondar:
        # O "+ 0.0" evita a impressão de "-0.0" quando o valor é um zero negativo.
        return round(x, 1) + 0.0, round(y, 1) + 0.0
    return x, y


def transforma_ponto(ponto, theta1, theta2, l1=L1, l2=L2, graus=True):
    """Converte um ponto dado no referencial do efetuador para (X_U, Y_U).

    Exemplo de uso: uma ferramenta presa 5 cm à frente da ponta do robô está no
    ponto (5, 0) do referencial do efetuador; esta função diz onde ela se
    encontra no referencial da base.
    """
    ponto = np.asarray(ponto, dtype=float).ravel()
    if ponto.size != 2:
        raise ValueError(f"O ponto deve ter 2 coordenadas; recebido {ponto.size}.")

    # Coordenadas homogêneas: acrescenta-se um 1 para que a translação atue.
    homogeneo = np.array([ponto[0], ponto[1], 1.0])
    resultado = matriz_transformacao(theta1, theta2, l1, l2, graus) @ homogeneo
    return float(resultado[0]), float(resultado[1])


# ---------------------------------------------------------------------------
# Demonstração
# ---------------------------------------------------------------------------
def _titulo(texto):
    print()
    print("=" * 72)
    print(texto)
    print("=" * 72)


def main():
    np.set_printoptions(precision=4, suppress=True)

    print(f"Robô planar de dois elos:  L1 = {L1:.1f} cm,  L2 = {L2:.1f} cm")

    # ---- (a) posição do efetuador em algumas configurações -----------------
    _titulo("Item (a) - Posição do efetuador final")
    casos = [
        (0.0, 0.0, "braço totalmente estendido sobre o eixo X_U"),
        (90.0, 0.0, "braço estendido na vertical"),
        (0.0, 90.0, "elo 1 na horizontal, elo 2 apontando para cima"),
        (30.0, 45.0, "configuração genérica"),
        (45.0, -45.0, "theta2 negativo (sentido horário)"),
        (0.0, 180.0, "elo 2 dobrado sobre o elo 1"),
        (180.0, 0.0, "braço estendido no sentido -X_U"),
    ]
    print(f"{'theta1':>8} {'theta2':>8} {'X_U (cm)':>10} {'Y_U (cm)':>10}   descrição")
    print("-" * 72)
    for t1, t2, descricao in casos:
        x, y = posicao_efetuador(t1, t2)
        print(f"{t1:8.1f} {t2:8.1f} {x:10.1f} {y:10.1f}   {descricao}")

    # ---- (b) matriz de transformação ---------------------------------------
    _titulo("Item (b) - Matriz de transformação (efetuador -> X_U, Y_U)")
    theta1, theta2 = 30.0, 45.0
    T = matriz_transformacao(theta1, theta2)
    print(f"Para theta1 = {theta1} graus e theta2 = {theta2} graus:")
    print()
    print("T =")
    print(T)
    print()
    print("Leitura da matriz:")
    print(f"  orientação do efetuador  : theta1 + theta2 = {theta1 + theta2} graus")
    print(f"  posição (última coluna)  : X_U = {T[0, 2]:.1f} cm,  Y_U = {T[1, 2]:.1f} cm")

    # ---- Verificações -------------------------------------------------------
    _titulo("Verificações")

    # 1. A forma fechada coincide com o produto das matrizes elementares.
    t1r, t2r = np.radians(theta1), np.radians(theta2)
    c12, s12 = np.cos(t1r + t2r), np.sin(t1r + t2r)
    T_fechada = np.array([[c12, -s12, L1 * np.cos(t1r) + L2 * c12],
                          [s12, c12, L1 * np.sin(t1r) + L2 * s12],
                          [0.0, 0.0, 1.0]])
    print(f"1) Forma fechada x produto R.D.R.D : diferença máxima = "
          f"{np.max(np.abs(T - T_fechada)):.2e}")

    # 2. O bloco de rotação é ortogonal e preserva distâncias (det = +1).
    R = T[:2, :2]
    print(f"2) Bloco de rotação ortogonal      : ||R^T R - I|| = "
          f"{np.linalg.norm(R.T @ R - np.eye(2)):.2e},  det(R) = {np.linalg.det(R):.6f}")

    # 3. A origem do referencial do efetuador cai sobre a posição do efetuador.
    origem = transforma_ponto([0.0, 0.0], theta1, theta2)
    posicao = posicao_efetuador(theta1, theta2, arredondar=False)
    print(f"3) Origem do ref. do efetuador     : {np.round(origem, 4)} "
          f"= posição do efetuador {np.round(posicao, 4)}")

    # 4. O alcance do robô nunca ultrapassa L1 + L2 nem fica aquém de |L1 - L2|.
    gerador = np.random.default_rng(7)
    angulos = gerador.uniform(-360.0, 360.0, size=(2000, 2))
    distancias = np.array([np.hypot(*posicao_efetuador(a, b, arredondar=False))
                           for a, b in angulos])
    print(f"4) Alcance em 2000 sorteios        : mínimo = {distancias.min():.2f} cm, "
          f"máximo = {distancias.max():.2f} cm  (esperado entre "
          f"{abs(L1 - L2):.1f} e {L1 + L2:.1f})")

    # 5. Ângulos em graus e em radianos devem dar o mesmo resultado.
    em_graus = posicao_efetuador(30.0, 45.0, arredondar=False)
    em_radianos = posicao_efetuador(np.pi / 6, np.pi / 4, graus=False, arredondar=False)
    print(f"5) Graus x radianos                : diferença máxima = "
          f"{np.max(np.abs(np.array(em_graus) - np.array(em_radianos))):.2e}")

    # ---- Uso prático da matriz ---------------------------------------------
    _titulo("Uso da matriz: ferramenta 5 cm à frente da ponta do robô")
    ferramenta = transforma_ponto([5.0, 0.0], theta1, theta2)
    print(f"Ponto (5,0) no referencial do efetuador  ->  "
          f"({ferramenta[0]:.1f}, {ferramenta[1]:.1f}) cm no referencial da base.")
    print("Como o elo 2 mede 15 cm, esse ponto fica a 20 cm da segunda junta,")
    print("na mesma direção do elo 2.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass
    main()
