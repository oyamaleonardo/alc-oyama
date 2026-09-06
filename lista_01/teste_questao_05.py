# -*- coding: utf-8 -*-
r"""
Testes da 5a Questão (robô planar de dois elos).

Este arquivo NÃO faz parte da resposta da lista: ele existe só para conferir se
o `questao_05.py` está funcionando.

Ao final aparece um resumo dizendo quantos testes passaram.
Para testar com os seus próprios ângulos, vá até a seção "MEXA AQUI".
"""

import sys

import numpy as np

from questao_05 import (L1, L2, matriz_transformacao, posicao_efetuador,
                        transforma_ponto)

# Faz o terminal do Windows exibir os acentos corretamente.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except (AttributeError, OSError):
    pass

# Contadores usados no resumo final.
_passou = 0
_falhou = 0


def verifica(descricao, obtido, esperado, tolerancia=1e-10):
    """Confere se `obtido` é (numericamente) igual a `esperado` e imprime o resultado."""
    global _passou, _falhou

    obtido = np.asarray(obtido, dtype=float).ravel()
    esperado = np.asarray(esperado, dtype=float).ravel()

    if obtido.shape == esperado.shape and np.all(np.abs(obtido - esperado) <= tolerancia):
        _passou += 1
        print(f"  [OK]    {descricao}")
        print(f"          resultado = {np.round(obtido, 4)}")
    else:
        _falhou += 1
        print(f"  [FALHA] {descricao}")
        print(f"          esperado  = {np.round(esperado, 4)}")
        print(f"          obtido    = {np.round(obtido, 4)}")


def verifica_erro(descricao, tipo_de_erro, funcao):
    """Confere que `funcao()` levanta o erro esperado."""
    global _passou, _falhou

    try:
        resultado = funcao()
    except tipo_de_erro as erro:
        _passou += 1
        print(f"  [OK]    {descricao}")
        print(f"          {tipo_de_erro.__name__}: {erro}")
    except Exception as erro:  # levantou, mas não foi o erro que esperávamos
        _falhou += 1
        print(f"  [FALHA] {descricao}")
        print(f"          esperava {tipo_de_erro.__name__}, veio {type(erro).__name__}: {erro}")
    else:
        _falhou += 1
        print(f"  [FALHA] {descricao}")
        print(f"          nenhum erro foi levantado; retornou {resultado}")


def titulo(texto):
    print()
    print(texto)
    print("-" * len(texto))


# ===========================================================================
# ITEM (a) - POSIÇÃO DO EFETUADOR
# ===========================================================================

titulo("1) Posições que dá para conferir só olhando o desenho")

# Braço todo esticado sobre o eixo X: a ponta fica em 20 + 15 = 35 cm.
verifica("theta1=0, theta2=0  ->  (35.0, 0.0)",
         posicao_efetuador(0, 0), [35.0, 0.0])

# Mesma coisa, mas girado 90 graus: a ponta sobe para 35 cm de altura.
verifica("theta1=90, theta2=0  ->  (0.0, 35.0)",
         posicao_efetuador(90, 0), [0.0, 35.0])

# Elo 1 deitado (20 cm em X) e elo 2 em pé (15 cm em Y): forma um "L".
verifica("theta1=0, theta2=90  ->  (20.0, 15.0)",
         posicao_efetuador(0, 90), [20.0, 15.0])

# Elo 2 dobrado para trás sobre o elo 1: sobra 20 - 15 = 5 cm.
verifica("theta1=0, theta2=180  ->  (5.0, 0.0)",
         posicao_efetuador(0, 180), [5.0, 0.0])

# Braço esticado apontando para o lado negativo do eixo X.
verifica("theta1=180, theta2=0  ->  (-35.0, 0.0)",
         posicao_efetuador(180, 0), [-35.0, 0.0])


titulo("2) Configuração genérica, conferida com a fórmula na mão")

# X = 20*cos(30) + 15*cos(75) = 17.3205 + 3.8823 = 21.2028  -> 21.2
# Y = 20*sen(30) + 15*sen(75) = 10.0000 + 14.4889 = 24.4889 -> 24.5
verifica("theta1=30, theta2=45  ->  (21.2, 24.5)",
         posicao_efetuador(30, 45), [21.2, 24.5])

# theta2 negativo (junta girada no sentido horário):
# X = 20*cos(45) + 15*cos(0) = 14.1421 + 15 = 29.1421 -> 29.1
# Y = 20*sen(45) + 15*sen(0) = 14.1421 + 0  = 14.1421 -> 14.1
verifica("theta1=45, theta2=-45  ->  (29.1, 14.1)",
         posicao_efetuador(45, -45), [29.1, 14.1])


titulo("3) O enunciado pede 1 casa decimal")

x, y = posicao_efetuador(33.7, 12.9)
casas_ok = (round(x, 1) == x) and (round(y, 1) == y)
verifica(f"resultado ({x}, {y}) tem no máximo 1 casa decimal",
         [1.0 if casas_ok else 0.0], [1.0])


titulo("4) Girar uma volta inteira (360 graus) não muda nada")

verifica("theta1=30+360, theta2=45  ->  igual a theta1=30, theta2=45",
         posicao_efetuador(390, 45), posicao_efetuador(30, 45))


titulo("5) Graus e radianos devem concordar")

verifica("30 graus = pi/6 rad, 45 graus = pi/4 rad",
         posicao_efetuador(np.pi / 6, np.pi / 4, graus=False, arredondar=False),
         posicao_efetuador(30, 45, arredondar=False))


titulo("6) O robô não alcança mais que L1+L2 nem menos que |L1-L2|")

gerador = np.random.default_rng(123)
distancias = [np.hypot(*posicao_efetuador(a, b, arredondar=False))
              for a, b in gerador.uniform(-360, 360, size=(500, 2))]
dentro = all(abs(L1 - L2) - 1e-9 <= d <= L1 + L2 + 1e-9 for d in distancias)
verifica(f"500 sorteios entre {abs(L1 - L2):.1f} e {L1 + L2:.1f} cm "
         f"(obtido: {min(distancias):.2f} a {max(distancias):.2f})",
         [1.0 if dentro else 0.0], [1.0])


# ===========================================================================
# ITEM (b) - MATRIZ DE TRANSFORMAÇÃO
# ===========================================================================

titulo("7) Formato da matriz")

T = matriz_transformacao(30, 45)
verifica("a matriz é 3x3", list(T.shape), [3.0, 3.0])
verifica("a última linha é [0, 0, 1]", T[2, :], [0.0, 0.0, 1.0])


titulo("8) Com os dois ângulos zerados, sobra só a translação de 35 cm")

# Sem rotação nenhuma, a matriz vira a identidade com um deslocamento em X.
verifica("T(0, 0) = identidade + translação de 35 em X",
         matriz_transformacao(0, 0),
         [[1.0, 0.0, 35.0],
          [0.0, 1.0, 0.0],
          [0.0, 0.0, 1.0]])


titulo("9) A última coluna da matriz é a posição do efetuador (item a)")

T = matriz_transformacao(30, 45)
verifica("T[0:2, 2] = posição calculada pela função do item (a)",
         T[:2, 2], posicao_efetuador(30, 45, arredondar=False))


titulo("10) O bloco 2x2 é uma rotação de verdade")

# Uma matriz de rotação satisfaz R^T . R = I  e  det(R) = +1.
R = matriz_transformacao(30, 45)[:2, :2]
verifica("R^T . R = identidade", R.T @ R, np.eye(2))
verifica("det(R) = 1", [np.linalg.det(R)], [1.0])

# E a orientação do efetuador é theta1 + theta2 = 75 graus.
angulo = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
verifica("a orientação do efetuador é 75 graus", [angulo], [75.0])


titulo("11) A transformação preserva distâncias (é um movimento rígido)")

# Dois pontos quaisquer no referencial do efetuador...
p, q = [2.0, 3.0], [-1.0, 4.0]
distancia_antes = np.hypot(p[0] - q[0], p[1] - q[1])
# ...continuam à mesma distância depois de levados para o referencial da base.
P = np.array(transforma_ponto(p, 37, -22))
Q = np.array(transforma_ponto(q, 37, -22))
verifica("a distância entre dois pontos não muda",
         [np.linalg.norm(P - Q)], [distancia_antes])


titulo("12) A origem do referencial do efetuador é a ponta do robô")

verifica("o ponto (0,0) do efetuador vira a posição do efetuador",
         transforma_ponto([0, 0], 30, 45),
         posicao_efetuador(30, 45, arredondar=False))


titulo("13) Um ponto à frente da ponta cai na direção do elo 2")

# O ponto (5,0) do referencial do efetuador está 5 cm adiante da ponta, logo a
# 15 + 5 = 20 cm da segunda junta. Com theta1=0 e theta2=0, tudo fica no eixo X:
# a segunda junta está em x=20, então o ponto cai em x=40.
verifica("com o braço esticado, o ponto (5,0) cai em (40.0, 0.0)",
         transforma_ponto([5, 0], 0, 0), [40.0, 0.0])


titulo("14) Entrada inválida deve dar erro")

verifica_erro("ponto com 3 coordenadas em vez de 2", ValueError,
              lambda: transforma_ponto([1, 2, 3], 30, 45))


# ===========================================================================
# MEXA AQUI: escreva os seus próprios testes
# ===========================================================================
titulo("15) Seus testes")

# Escolha os ângulos das duas juntas (em graus):
meu_theta1 = 60.0
meu_theta2 = 30.0

# Calcule na mão o resultado que você espera:
#   X = 20*cos(60) + 15*cos(90) = 10.0 + 0.0  = 10.0
#   Y = 20*sen(60) + 15*sen(90) = 17.32 + 15  = 32.3
meu_esperado = [10.0, 32.3]

verifica(f"meu teste: theta1={meu_theta1}, theta2={meu_theta2}",
         posicao_efetuador(meu_theta1, meu_theta2), meu_esperado)


# ===========================================================================
# RESUMO
# ===========================================================================
print()
print("=" * 60)
print(f"RESUMO: {_passou} passaram, {_falhou} falharam")
print("=" * 60)
if _falhou == 0:
    print("Tudo certo!")
else:
    print("Algum teste falhou - veja as linhas marcadas com [FALHA] acima.")
