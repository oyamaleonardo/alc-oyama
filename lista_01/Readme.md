# Questão 4

Desenvolvido com ajuda do Claude Code, modelo `Opus 5.0`.

## questao_04.py

| Bloco	| Finalidade | Resultado Esperado |
| --- | --- | --- |
| Caso 1 | Resolve um sistema 3×3 | `Resíduo ... = 0.0` |
| Caso 2 | Bate com o SciPy num 6×6 aleatório | `Diferença máxima = 0.0` |
| Caso 3 | Aceita vetor simples, não só coluna | imprime `[0.75 2.5 2.]` |
| Caso 4 | A exception do enunciado | `DiagonalNulaError levantada com sucesso` |
| Caso 5 | Rejeita tamanhos incompatíveis | `ValueError levantada com sucesso` |

## teste_questao_04.py

Cada teste imprime `[OK]` ou `[FALHA]`, e no fim vem o resumo `12 passaram, 0 falharam`.

**O que os testes cobrem**

1. Sistema 2×2 e 3×3 com contas que você consegue conferir na mão (os comentários mostram o cálculo)
2. Matriz identidade — a resposta tem que ser o próprio `b`
3. Matriz 1×1, o menor caso possível
4. Vetor escrito de forma simples (`[4, 9]`) em vez de coluna
5. Teste "de trás para frente": sorteia 20 matrizes, escolhe a resposta `x`, calcula `b = U·x` e verifica se a função recupera o `x` original — erro máximo deu `8.9e-15`, ou seja, zero na prática
6. Zero na diagonal, incluindo um valor minúsculo `(1e-20)` que não é zero exato mas é zero na prática — os dois devem dar erro
7. Entradas inválidas: matriz não quadrada, `b` de tamanho errado, matriz que não é triangular superior

**A seção "MEXA AQUI"** (item 9, perto do fim do arquivo) serve para testes customizados: tem uma matriz, um vetor e o resultado esperado. 

Troque os números, salve, rode de novo. Se a conta estiver errada, vai aparecer `[FALHA]` mostrando lado a lado o esperado e o obtido.