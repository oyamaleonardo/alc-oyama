# Questão 4

Desenvolvida com ajuda do Claude Code, modelo `Opus 5.5`.

## questao_04.py

Investiga a matriz `A = u vᵀ` para `u` e `v` sorteados (`rand(n,1)`), com n = 5, 15 e 25.

Para rodar com resultados reprodutíveis, passe uma semente:

```
python questao_04.py 42
```

**Conclusões** (para u e v não nulos):

1. `posto(u vᵀ) = 1`
2. `‖u vᵀ‖₂ = ‖u‖₂ ‖v‖₂`

| Bloco | Finalidade | Resultado esperado |
| --- | --- | --- |
| Experimento do enunciado | Posto e normas para n = 5, 15, 25 | posto `1`; as duas normas iguais |
| Afirmação 1, passos 2 e 3 | Núcleo de A = vetores ortogonais a `v` | `dim N(A) = n - 1`; `max \|vᵀx\|` ~ 1e-16 |
| Afirmação 1, passo 4 | posto = n − nulidade | `n - nulidade = 1 = matrix_rank` |
| Afirmação 1, prova alternativa | Toda coluna j de A é `vⱼ·u` | resíduo ~ 1e-16 |
| Afirmação 2, passos 1 e 2 | `‖Ax‖ ≤ ‖u‖‖v‖` para x unitário | máximo sorteado abaixo da cota |
| Afirmação 2, passo 3 | A cota é atingida em `x = v/‖v‖` | diferença ~ 1e-16 |
| Valores singulares e posto numérico | Só σ₁ é não nulo; demais abaixo da tolerância | `σ₁ = ‖u‖‖v‖`; demais ~ 1e-16 |
| Caso n = 5 por extenso | Mostra u, v e A | — |

As demonstrações estão na solução escrita da lista.

# Questão 5

Desenvolvida com ajuda do Claude Code, modelo `Opus 5.5`.

## questao_05.py

Verifica se uma matriz quadrada é ortogonal de duas formas. As duas funções recebem a matriz e retornam um booleano.

| Função | Critério |
| --- | --- |
| `is_orthogonal_by_definition(P)` | `PᵀP = I` |
| `is_orthogonal_by_vectors(P)` | colunas com norma 1 e ortogonais duas a duas |

As matrizes do enunciado têm 5 algarismos significativos, por isso a comparação usa a tolerância `tol = 1e-4`.

| Matriz | Resultado esperado |
| --- | --- |
| 6.38 a) | `True` nas duas funções |
| 6.38 b) | `True` nas duas funções |
| 6.39 a) | `True` nas duas funções |
| 6.39 b) | `False` nas duas funções: as colunas têm norma 1, mas não são ortogonais |
