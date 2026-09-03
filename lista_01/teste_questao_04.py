# -*- coding: utf-8 -*-
r"""
Testes da 4a Questão (substituição regressiva).

Este arquivo NÃO faz parte da resposta da lista: ele existe só para conferir se
o `questao_04.py` está funcionando.

Ao final aparece um resumo dizendo quantos testes passaram.
Para testar com os seus próprios números, vá até a seção "MEXA AQUI".
"""

import sys

import numpy as np

from questao_04 import DiagonalNulaError, substituicao_regressiva

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
        print(f"          resultado = {obtido}")
    else:
        _falhou += 1
        print(f"  [FALHA] {descricao}")
        print(f"          esperado  = {esperado}")
        print(f"          obtido    = {obtido}")


def verifica_erro(descricao, tipo_de_erro, funcao):
    """Confere que `funcao()` levanta o erro esperado.

    Aqui o erro é o comportamento CORRETO: o enunciado pede que a implementação
    levante uma exception quando houver um zero na diagonal.
    """
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
        print(f"          nenhum erro foi levantado; retornou {np.asarray(resultado).ravel()}")


def titulo(texto):
    print()
    print(texto)
    print("-" * len(texto))


# ===========================================================================
# TESTES
# ===========================================================================

titulo("1) Sistema 2x2 resolvido na mão")
# 2x + 1y = 4
#      3y = 9   ->  y = 3  ->  2x + 3 = 4  ->  x = 0.5
U = np.array([[2.0, 1.0],
              [0.0, 3.0]])
b = np.array([[4.0],
              [9.0]])
verifica("x deve ser [0.5, 3.0]", substituicao_regressiva(U, b), [0.5, 3.0])


titulo("2) Sistema 3x3 resolvido na mão")
#  1x + 2y + 3z = 14
#       1y + 1z =  5   ->  z = 2, y = 3, x = 14 - 6 - 6 = 2
#            1z =  2
U = np.array([[1.0, 2.0, 3.0],
              [0.0, 1.0, 1.0],
              [0.0, 0.0, 1.0]])
b = np.array([[14.0],
              [5.0],
              [2.0]])
verifica("x deve ser [2.0, 3.0, 2.0]", substituicao_regressiva(U, b), [2.0, 3.0, 2.0])


titulo("3) Matriz identidade: a resposta é o próprio b")
U = np.eye(4)
b = np.array([[7.0], [-1.0], [0.5], [3.0]])
verifica("x deve ser igual a b", substituicao_regressiva(U, b), b)


titulo("4) Matriz 1x1 (o menor caso possível)")
verifica("5x = 10  ->  x = 2", substituicao_regressiva([[5.0]], [[10.0]]), [2.0])


titulo("5) Aceita o vetor escrito de forma simples, sem ser coluna")
U = np.array([[2.0, 1.0],
              [0.0, 3.0]])
verifica("mesmo resultado com b = [4, 9]", substituicao_regressiva(U, [4.0, 9.0]), [0.5, 3.0])


titulo("6) Teste 'de trás para frente' com 20 matrizes aleatórias")
# Ideia: sorteamos a resposta x, calculamos b = U @ x e conferimos se a função
# consegue recuperar o x original a partir de U e b.
gerador = np.random.default_rng(0)
erro_maximo = 0.0
for _ in range(20):
    n = int(gerador.integers(2, 9))
    U = np.triu(gerador.uniform(-5.0, 5.0, size=(n, n)))
    U[np.diag_indices(n)] += 6.0            # garante diagonal longe de zero
    x_verdadeiro = gerador.uniform(-10.0, 10.0, size=(n, 1))
    b = U @ x_verdadeiro
    x_calculado = substituicao_regressiva(U, b)
    erro_maximo = max(erro_maximo, float(np.max(np.abs(x_calculado - x_verdadeiro))))
verifica(f"erro máximo nas 20 matrizes (~0)", [erro_maximo], [0.0], tolerancia=1e-8)


titulo("7) Zero na diagonal DEVE dar erro (é o que o enunciado pede)")
U = np.array([[1.0, 2.0, 3.0],
              [0.0, 0.0, 4.0],
              [0.0, 0.0, 5.0]])
b = np.array([[1.0], [2.0], [3.0]])
verifica_erro("zero no meio da diagonal", DiagonalNulaError,
              lambda: substituicao_regressiva(U, b))

U = np.array([[1.0, 2.0],
              [0.0, 1e-20]])          # não é zero exato, mas é zero na prática
verifica_erro("valor minúsculo na diagonal (1e-20)", DiagonalNulaError,
              lambda: substituicao_regressiva(U, [1.0, 1.0]))


titulo("8) Entradas inválidas também devem dar erro")
verifica_erro("matriz não quadrada (2x3)", ValueError,
              lambda: substituicao_regressiva(np.ones((2, 3)), [1.0, 1.0]))

verifica_erro("b com tamanho errado", ValueError,
              lambda: substituicao_regressiva(np.eye(3), [[1.0], [2.0]]))

verifica_erro("matriz não é triangular superior", ValueError,
              lambda: substituicao_regressiva(np.array([[1.0, 2.0],
                                                        [3.0, 4.0]]), [1.0, 1.0]))


# ===========================================================================
# MEXA AQUI: escreva os seus próprios testes
# ===========================================================================
titulo("9) Seus testes")

# Monte a matriz triangular superior (só números acima e na diagonal):
meu_U = np.array([[4.0, -2.0],
                  [0.0,  1.0]])

# Monte o vetor coluna:
meu_b = np.array([[6.0],
                  [3.0]])

# Escreva abaixo o resultado que você espera (calcule na mão para conferir):
meu_esperado = [3.0, 3.0]

verifica("meu teste", substituicao_regressiva(meu_U, meu_b), meu_esperado)


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
