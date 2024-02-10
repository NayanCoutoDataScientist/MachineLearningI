# Modelos de aprendizagem de máquina
"""
Modelos paramétricos e não paramétricos

Testes Paramétricos:
> Testes paramétricos exigem para sua utilização que seja pressuposto uma distribuição dos dados (na saúde, a maioria
exige pressuposto da distribuição normal.
> É um teste estatístico, no qual suposições específicas são feitas sobre o parâmetro da população, é conhecido como
teste paramétrico.

Estatística: Distribuição
Medição: Intervalo ou razão
Tendência: Significar
População: Completamente conhecida
Ex: Teste-t Pareado, Teste-t Não Pareado, Correlação de Pearson, Anova.

Teste Não-paramétrico:
> Testes não paramétricos não requerem pressuposto da distribuição dos dados.
> É um teste estatístico usado no caso de variáveis independentes não-métricas. é chamado de teste não paramétrico.

Estatística: Arbitrário
Medição: Nominal ou ordinal
Tendência: Mediana
População: Indisponível
Ex: Teste da soma de Wilcoxon Rank, Teste de Mann-Whitney, Correlação de Spearman, Teste de Kruskal Wallis.
"""

# Método não paramétrico – Método de Kernel
"""
Mapas de Calor de Kernel
> Neste método cada uma das observações é ponderada pela distância em relação a um valor central, o núcleo.
> O Mapa de Kernel é uma alternativa para análise geográfica do comportamento de padrões. No mapa é plotado, por meio
de métodos de interpolação, a intensidade pontual de determinado fenômeno em toda a região de estudo.
"""

# Modelo paramétrico – Regressão Linear
"""
Linear Regression
> Regressão linear é o processo de traçar uma reta através dos dados em um diagrama de dispersão. A reta resume esses
dados, o que é útil quando fazemos previsões.

> Variável independente: aqui, temos um valor que vai influenciar, diretamente, 
o que se deseja encontrar.
> Variável dependente: essa variável é aquela que desejamos prever, ou seja, ela depende de outros
valores.

> Considerando a regressão linear, temos dois tipos principais:
- Na regressão linear simples temos somente uma variável independente,
que é utilizada para fazer uma determinada predição;
Fórmula: f(x) = a+bx

- Na regressão linear múltipla estamos falando sobre a existência de várias variáveis independentes
que são utilizadas para a realização da predição, consequentemente,
a forma de apresentação dessas regressões depende do seu tipo, sendo simplesmente uma reta ou,
até mesmo, um plano que leva duas ou mais dimensões.
Fórmula: f(x) = a + b_i*x_i + b_ii*x_ii + ... + b_n*x_n
"""

# Estudo de Caso:

import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Dados de vendas
vendas = np.array([-500, -420, -300]).reshape(-1, 1)
meses = np.array([6, 7, 8]).reshape(-1, 1)

# Criar o modelo de regressão linear
modelo = LinearRegression()

# Ajustar o modelo aos dados de vendas
modelo.fit(vendas, meses)

# Fazer uma previsão com base no valor de x = 0
x = np.array([[-300]])
previsao = modelo.predict(x)

print("Previsão de vendas:", previsao)

# Plotar o gráfico de dispersão
plt.scatter(vendas, meses, color='blue')
plt.xlabel('Vendas')
plt.ylabel('Meses')
plt.title('Gráfico de Dispersão - Vendas')
plt.show()

"""
Nesse caso, temos como equação: f(a) = -500 + 80x (em que -500 representa o valor gasto no produto e 80 representa
o lucro unitário). Considerando que x está relacionado à quantidade a ser vendida, verifica-se que, a partir da venda
de 7 produtos, a empresa começará a ter lucro, cobrindo os gastos iniciais.
"""