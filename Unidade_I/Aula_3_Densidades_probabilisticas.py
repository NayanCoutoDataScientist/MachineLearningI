# Fundamentos de probabilidade
"""
> Analisa a possibilidade de um fato ocorrer.

Probabilidade:
> Espaço Amostral - conjunto de todos os resultados possíveis.
> Eventos - Subconjuntos de espaços amostrais.
> População.
> Amostra.
"""

# Variável aleatória discreta
"""
> Funçaõ definida sobre o espaço amostral que atribui um determinado valor real a cada elemento desse espaço amostral.

Exemplos:
> Número de chamada na policia no período da tarde;
>Número de alunos aprovados em uma disciplina com 70 alunos;
"""

# Variável aleatória contínua
"""
> Pode assumir qualquer valor numérico dentro de um intervalo ou de uma série de intervalos.

Exemplos:
> Pesos de animais.
> Tempo de falha de equipamento eletrônico.
> Retorno financeiro de investimentos.
"""

# Estudo de Caso:
"""
“Qual é a probabilidade da pessoa escolhida aleatoriamente ser homem?
E qual é a probabilidade de ser mulher?
"""

# Dicionário dos Dados
homens = {"classe A": 13059, "classe B": 3148, "classe C": 4679}
mulheres = {"classe A": 2641, "classe B": 2469, "classe C": 1393}

# Encontra o Total
total_homens = sum(homens.values())
total_mulheres = sum(mulheres.values())

# Encontra a probabilidade
probabilidade_homem = total_homens / (total_homens + total_mulheres)
probabilidade_mulher = total_mulheres / (total_homens + total_mulheres)

# Imprime a probabilidade
print(f"Probabilidade de ser homem: {probabilidade_homem:.2f}")
print(f"Probabilidade de ser mulher: {probabilidade_mulher:.2f}")