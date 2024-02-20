# Introdução à distribuição probabilística
"""
Probabolidade
> A probabilidade é uma medida de incerteza de vários fenômenos. Por exemplo, se você joga um dado, os resultados
possíveis são definidos pela probabilidade. Esta distribuição pode ser definida com quaisquer experimentos aleatórios,
cujo resultado não é certo ou não pode ser previsto.

Distribuição Probabilística
> Neste contexto, a distribuição de probabilidade produz os resultados possíveis para qualquer evento aleatório.
Também é definido com base no espaço amostral subjacente como um conjunto de resultados possíveis de qualquer
experimento aleatório. Essas configurações podem ser um conjunto de números reais, um conjunto de vetores ou
um conjunto de quaisquer entidades. É uma parte da probabilidade e da estatística.
> p(x) = a probabilidade de que a variável aleatória assuma um valor específico de x.
A soma de todas as probabilidades para todos os valores possíveis, deve ser igual a 1. Além disso, a probabilidade de
um determinado valor ou intervalo de valores deve estar entre 0 e 1.
"""

# Distribuições Discretas e Contínuas
"""
Distribuição Discreta
> Para funções de distribuição de probabilidade discreta, cada valor possível tem uma probabilidade diferente de zero.
Além disso, as probabilidade para todos os valores possíveis devem somar um. Como a probabilidade total é 1, um dos
valores deve ocorrer para cada oportunidade.
> Por exemplo, a probabilidade de ter um número em um dado é de 1/6. A probabilidade total para todos os seis valores
é igual a um. Ao rolar um dado, você inevitávelmente obtém um dos valores possíveis.

> Distribuição binomial para modelar dados binários, como lançamentos de moedas.
      p_xk = (n/k) * (P**k)*(Q**(n-1))
> Distribuição de Poisson par modelar dados de contagem, como a contagem de retiradas de livros da biblioteca por hora.
      p_x = ((l**x) * (math.e**(-l))) / math.factorial(x)
> Distribuição uniforme para modelar vários eventos com a mesma probabilidade, como lançar um dado.
      1/(b-a)

Distribuição Contínua
> Já nas distribuições contínuas, são conhecidas como funções de densidade de probabilidade. Sabe-se que tem uma
distribuição contínua se a variável pode assumir um número infinito de valores entre quaisquer dois valores. Variáveis
contínuas geralmente são medidas em uma escala, como altura, peso e temperatura.
"""

# Distribuições de Probabilidade Conjunta
"""
> Podemos estar interessados na probabilidade de dois eventos simultâneos, por exemplo, os resultados de duas variáveis
aleatórias diferentes.
> A probabilidade de dois (ou mais) eventos é chamada de probabilidade conjunta. A probabilidade conjunta de duas ou
mais variáveis aleatórias é conhecida como distribuição de probabilidade conjunta.
> Por exemplo, a probabilidade conjunta do evento A e do evento B é escrita formalmente como: 
      P(A e B)

Distribuição Discreta
> O 'e' ou conjunção é denotado usando o operador 'U' maiúsculo de cabeça para baixo '^', ou as vezes uma vírgula ','.
      P(A^B)
      P(A,B)
> A probabilidade conjunta para os eventos A e B é calculada como a probabilidade do evento A dado evento B multiplicado
pela probabilidade do evento B. Isso pode ser declarado formalmente da seguinte forma:
      P(A e B) = P(A dado B) * P(B)
"""

# Estudo de Caso
"""
(...) Após identificar, vamos iniciar a análise. O departamento de vendas da empresa identificou que, em média,
há 5 clientes por hora. Esses clientes compram o produto oferecido pela empresa.
Os gestores querem saber qual é a probabilidade de receber mais dois clientes em uma hora selecionada de forma aleatória.
"""

import math

x = 2
l = 5

p_x = ((l**x) * (math.e**(-l))) / math.factorial(x)
p_x100 = p_x * 100

print(f'O resultado da distribuição de probabilidade é {p_x:.5f}, ou {p_x100:.2f}'
      f'% de chance de receber clientes em uma hora selecionada de forma aleatória.')