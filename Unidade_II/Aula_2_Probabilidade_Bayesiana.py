# Distribuições a Priori
"""
Na Inferência Bayesiana:
> Utilização de INFORMAÇÃO A PRIORI;
> Necessidade da especificação de uma DISTRIBUIÇÃO A PRIORI;

Conceitos Importantes:
> Probabilidade
    - Medida de incerteza relacionada à ocorrência de um evento (ou observação de um valor).
> Distribuição de probabilidade
    - Parâmetro desconhecido

Parâmetros
> Aleatórios
> Distribuição anteriormente da observação da amostra.
    - Distribuição a priori.
> Toda a informação da amostra é representada pela função de verossimilhança.
> Distribuição para atualização dos parâmetros depois da observação dos elementos da amostra.
    - Distribuição a posteriori

Exemplo Jogo de Moedas:
> Antes de Jogar alguma vez?
> Após observar os resultaddos dos lançamentos 1, 2, 3?
> A priori: T(A)aA^a-1 (1-A)^b-1
> A posteriori: T(A|x)aA^x (1-A)^n-x xaA^a-1 (1-A)^b-1 = A^x+a-1 (1-A)^n-x+b-1

> Distribuição a posteriori coroa:
    T(A|x1 = 1)aA^0 (1-A)^0 xA^1 (1-A)^0 = A^1 (1-A)^0 = A

> Distribuição a posteriori cara:
    T(A|x1 = 0)aA^0 (1-A)^0 xA^0 (1-A)^1 = A^0 (1-A)^1 = 1-A
    T(A|x1, x2)aA^x1+x2 = (1-A)^2-x1-x2
    T(A|x1, x2, x3)aA^x1+x2+x3 = (1-A)^3-x1-x2-x3
"""

# Estimação
"""
> Processo de Aprendizagem:
    - Baseado em uma população.
> Inferência:
    - Realizada a partir de amostras.
> Estimação:
    - Variável aleatória - distribuição de probabilidade.
    - Distribuição de probabilidade - parâmetros populacionais.
    
Estimação:
> Estimação de parâmetros:
    - Uso de dados da amostra para estimar valores de parâmetros populacionais desconhecidos, como: média, dp e variância.
> Estimador:
    - Função, representado por uma variável aleatória por uma distribuição de probabilidade e seus parâmetros próprios.
> Estimativa:
    - Cada valor particular assumido por um estimador.

> Pontuais:
    - Valor obtido a partir dos resultados (dados) de uma variável aleatória de uma amostra representativa extraída da população.
    - Se uma amostra representativa da variável aleatória X é extraída da população,
    então a média e a variância podem ser utilizadas como estimadores pontuais.
> Por intervalos:
    - Geração de um intervalo, centrado na estimativa pontual, no qual se admite que esteja inserido o parâmetro populacional.
    - Calculada a partir de uma amostra extraída da população.
    - Geração de um intervalo de possíveis valores para o parâmetro populacional, a partir do valor encontrado na amostra.
    - Probabilidade (1-a).
    
    - Exemplo: intervalo de confiança de 95% (1-a = 0,95 e a=0,05). a+|- = Z_a/2 a/n^(1/2) -> Z_a/2 -> Z_0,0025 = 1,96
    - Bilateral: Xmed - Z_a/2 * a/n^(1/2) <= Med <= Xmed + Z_a/2 * a/n^(1/2)
    
    - Exemplo: 
        Variabilidade do tempo s = 0,10 minutos;
        n = 20 clientes;
        tempo médio = Xmed = 1,5;
        Intervalo de confiança = 95%;
        Valor do Intervalo = ?
        
        Xmed - Z_a/2 * a/n^(1/2) <= Med <= Xmed + Z_a/2 * a/n^(1/2)
        1,5 - 1,96 * 0,10/20^(1/2) <= Med <= 1,5 + 1,96 * 0,10/20^(1/2)
        1,46 <= Med <= 1,54
        
    - Erro de estimação: e = Z_a/2 * s/n^(1/2)
    - Tamanho da amostra: n = (Z_a/2 * s/e)^2
    
    - Exemplo:
        Desvio-padrão s=3;
        Confiança = 95%;
        Precisão = 0,5;
        n = ?
            a= 0,05
            Z_a/2 = 1,96
            s = 3,0
            e = 0,5
            n = (1,96 * (3,0/0,5))^2 = 138,3
"""

# Métodos aproximados
"""
Técnicas de Aproximação
> Monte Carlo

Método de Monte Carlo
> Estimação de valores futuros;
> Série de cálculos de probabilidade para estimar a chance de ocorrência deste evento.
> Resolução de problemas em diversas áreas, como: cálculo numérico, gestão de riscos, análise de mercado,
tomada de decisões, logística, otimização, etc.
> Suponha o cálculo da integral de uma função g(x) no intervalo entre a e b:
    I = Integral(B|A)g(x)dx
> Estimador:
    Î = (b-a) * 1/n[Somatória(n|i=1){g(x_i)}]
    
> Erro de estimação
    - Cálculo do erro entre o valor esperado e o valor estimado
    - Erro Quadrático Médio (EQM)
        EQM = 1/n * SUM(Î_i - I_i)^2
"""

# Estudo de Caso: Algoritmo de Monte Carlo

"""
Função: f(x) = (x^4) * (e^-x)
Iterações: 100
Intervalos: (1:5)
Integrar: I = Integral(5|1) * [(x^4) * (e^-x)] * dx
"""
import random
import math


def monte_carlo_integration(a, b, n):
    # Gerar amostras da distribuição U(a, b)
    samples = [random.uniform(a, b) for _ in range(n)]

    # Calcular g(x) para cada amostra
    g_values = [f(sample) for sample in samples]

    # Calcular a média da amostra
    sample_mean = sum(g_values) / n

    # Calcular a estimativa da integral
    integral_estimate = (b - a) * sample_mean

    return integral_estimate


# Função f(x) de exemplo
def f(x):
    return (x ** 4) * math.exp(-x)


# Parâmetros do algoritmo
a = 1  # Limite inferior
b = 5  # Limite superior
n = 100  # Número de amostras / iterações

# Executar o algoritmo de Monte Carlo
integral_estimate = monte_carlo_integration(a, b, n)

# Imprimir o resultado
print("Estimativa da integral:", integral_estimate)

# Valor esperado
expected_value = 13.34

# Calcular o erro quadrático médio (EQM)
eqm = ((integral_estimate - expected_value) ** 2) / n

# Imprimir o EQM
print("Erro Quadrático Médio (EQM):", eqm)