# Introdução aos métodos Bayesianos
"""
> Lojas virtuais e recomendações de produtos.
> Algoritmos baseados no Aprendizado de Máquina.
> Aprendizado baseado em dados e em experiências anteriores.
> Construção e treinamento de modelos estatísticos.

Estatística Bayesiana
> Inserção de conhecimentos que os dados originais não conseguem prever.
> Característica:
    - Não precisa que um determinado evento tenha ocorrido várias vezes para determinar a probabilidade de nova
    ocorrência.
    - Baseada na teoria de Bayes.
> Partes fundamentais:
    - Desconhecido: distribuições de probabilidade.
    - Conhecido: dados.
"""

# Teorema de Bayes
"""
    Suponha que um casal tenha dois filhos. Qual a probabilidade dos dois filhos serem meninas,
    considerando que um desses filhos é menina?
    
    Eventos:
        A (Evento desejado) - 2 meninas
        B (Evento realizado) - 1 menina
    
    Probabilidades:
        P(A) - 2 meninas
        P(B) - 1 meninas
        
    Binômio:
        (x+y)^n -> (x+y)² -> (x+y) * (x+y) -> x² + xy + xy + y² -> x² + 2xy + y²
        n -> número de filhos;
        x -> menina (50%)
        y -> menino (50%)
        
    Então:
        Duas meninas → x² → (1/2)² = 1/4
        Dois meninos → y² → (1/2)² = 1/4
        Uma menina e um menino → 2xy → 2 * 1/2 * 1/2 → 2/4
        Pelo menos uma menina → 1 - 1/4   = 3/4
        
    Logo
        P(A) = 1/4
        P(B) 3/4
        P(B|A) = 1
        P(A|B) = (1 * 1/4)/(3/4) = 1/3 
"""

# Teorema de Bayes e Probabilidade Condicional
"""
> Estatística Bayesiana:
    - Teorema de Bayes.
    - Cálculo da probabilidade de um evento, considerando que outro evento já ocorreu.

Teorema de Bayes
> Necessidade de informação anterior.
    - 'A priori';
    - O evento já ocorreu?
    - Qual sua probabilidade?
> Novas evidências para a obtenção de probabilidades posteriores (a posteriori).
> Basicamente, tem-se:
    - Variável desconhecida;
    - Objetivo principal: tentar reduzir este desconhecimento.
    
> Fórmula:
P(A|B) = [P(A) * P(B|A)] / [P(B)]
> Etapas Fundamentais:
    1) Reoresentar o desconhecimento 'a priori' com a distribuição de probabilidade a priori;
    2) Atualizar esse conhecimento com dados atuais (verossimilhança);
    3) Produzir uma distribuição de probabilidade para o parâmetro que possua menor
    desconhecimento (probabilidade a posteriori);
"""

# Princípio da Verossimilhança
"""
Verossimilhança
> Aquilo que parece intuitivamente verdadeiro;
> O que é atribuído a uma probabilidade de verdade, na relação ambígua que se estabelece entre imagem e idéia;
> Quantifica o quão plausível é uma hipótese ao serem consideradas todas as evidências ocorridas;

Para calcular a probabilidade de um paciente realmente ter câncer de pele, dado que o exame teve resultado positivo, 
podemos utilizar o Teorema de Bayes.
Vamos chamar o evento de ter câncer de pele de A e o evento de resultado positivo no exame de B

De acordo com o enunciado, temos as seguintes informações:
    P(A) = 0,01 (probabilidade de um paciente ter câncer de pele)
    P(B|A) = 0,8 (probabilidade de o exame detectar o câncer quando ele existe)
    P(B|¬A) = 0,096 (probabilidade de o exame detectar o câncer quando ele não existe)

Queremos calcular P(A|B), a probabilidade de um paciente ter câncer de pele dado que o exame teve resultado positivo.
Usando o Teorema de Bayes, temos:
    P(A|B) = (P(A) * P(B|A)) / P(B)

Para calcular P(B), podemos usar a lei da probabilidade total:
    P(B) = P(A) * P(B|A) + P(¬A) * P(B|¬A)

Substituindo os valores, temos:
    P(B) = 0,01 * 0,8 + 0,99 * 0,096

Agora podemos calcular P(A|B):
    P(A|B) = (0,01 * 0,8) / (0,01 * 0,8 + 0,99 * 0,096)
"""

P_A = 0.01
P_B_given_A = 0.8
P_B_given_not_A = 0.096

P_B = P_A * P_B_given_A + (1 - P_A) * P_B_given_not_A

P_A_given_B = (P_A * P_B_given_A) / P_B

print(f"A probabilidade de o paciente ter câncer de pele dado que o exame teve resultado positivo é: {P_A_given_B:.3f}")