# Introdução à teoria de decisão
"""
Contextualização

> Algorítmos de Aprendizado de Máquina
    - Tomam decisões a partir de informações ou experiências anteriores.
> Teoria da Decisão:
    - Escolhas de um agente.
    - Uso de processos racionais para selecionar a melhor alternativa dentre um conjunto de alternativas possíveis.
    - Teoria Bayesiana.

Teoria Bayesiana
Elementos:
> Classes Wi (variável aleatória)
> Probabilidades a priori P(Wi):
    - Decisões têm base em experiência anteriores.
    - Conhecimento anterior que se tem sobre um determinado problema (classes do problema).
> Probabilidade a posteriori P(Wi|X):
    - Decisões baseadas na agregação de novas informações à informação inicial (a priori).
    - Probabilidade de que um padrão pertença à classe Wi considerando a característica X.
> Função de densidade P(X|Wi):
    - Frequência de uma determinada característica com base em uma amostra.
> Regra de decisão:
    - W1, se P(W1) > P(W2);
    - W2, se P(W2) > P(W1);

Exemplo 2:
> Identificar a variável aleatória (Wi + tipo de peixe) e as probabilidades a priori (P(Wi)):
    - W_1: Tilápia;
    - W_2: Dourado;
    - P(W_1): Probabilidade a priori do tipo tilápia;
    - P(W_2): Probabilidade a priori do tipo dourado;
    - P(W_1) + P(W_2): 1;
> Regras de Decisão:
    - Decidir W1 se P(W_1) > P(W2);
    - Decidir W2 se P(W_2) > P(W1);
> Probabilidades:
    - P(W1) = 0,82;
    - P(W2) = 0,18;
> Característica intensidade luminosa do peixe:
    - Tilápias - 49,5% de intensidade clara;
    - Dourados - 85% de intensidade clara;
> Calcule a probabilidade do próximo peixe a ser pescado ser TILÁPIA, dado que o peixe pescado possuí intensidade clara.
> Se esta probabilidade for maior que 50%, então decida que o peixe é TILÁPIA;
> Resolução:
    P(W_1|W_2) = [P(W_1) * P(W_2|W_1)] / P(W_2) -> (0,82 * 0,495) / [(0,82 * 0,495) + (0,18 * 0,85)] = 0,726
"""

# Modelos de decisão
"""
> Teoria da Decisão:
    - Escolhas ed um agente.
    - Uso de processos racionais para selecionar a melhor alternativa dentre um conjunto de alternativas possíveis.
    - Modelos da Teoria Bayesiana:
        → Árvore de Decisão.
        
Árvore de Decisão
> Método estatístico.
> Aprendizagem supervisionada.
> Utilizado em problemas de classificação e na realização de previsões.
> A partir de um conjunto de dados existente, o método cria uma representação do conhecimento ali embutido,
em formato de árvore.
> Árvore de decisão binária:
    - Escolha SIM ou NÃO.
> Características:
    - Raiz: Contém o atributo mais representativo.
    - Folhas: Respostaso ou Decisões.
    - Cada Nó: Probabilidades ou escolhas.
"""

# Classificador Decision Tree
"""
Árvore de Decisão:
> Problemas de Decisão(classificação ou regressão);
> Formada por nós que contém escolhas ou decisões;
> Qual característica ou atributo será inserido no nó raíz?
> Quais serão os nós da subárvore da direita e da esquerda?
> Definição dos nós dessa árvore e em qual posição esses nós serão encaixados.
> Critérios: ganho de informação e entropia.
> Relacionados à desorganização e falta de uniformidade dos dados.

> Entropia:
    - Quanto maior, mais misturados e caóticos estão os dados.
    - Quanto menor, mais homogênea e uniforme estão os dados.

> Definição de posicionamento:
    - Calcular a entropia das classes e o ganho de informação dos atributos.
    - Aquele que tiver maior ganho de informação é definido como raiz da árvore.

> Definição das subárvores:
    - Novos cálculos de entropia e ganho com o conjunto de dados que atende à condição que leva à esquerda ou à direita.

> Divisão da base de dados:
    - Feita a partir de condições.

> Ganho de informação
    - Homogêneo - alto ganho de informação.

> Entropia
    - Homogeneidade.
    - Dados completamente homogêneos = entropia igual a 0
    - Dados divididos (50%/50%) = entropia igual a 1
    I(frq_maior, frq_menor) = -frqmaior log2 * frqmaior - frqmenor * log2 frqmenor
    
Exemplo:
> Duas classes envolvidas: 'Sim' e 'Não'
> Positivo: P(comprador = Sim) = 9/14
> Negativo: P(comprador = Não) = 5/14
> Entropia:
    - Entropia = 0: dados puros
    - Entropia = 1: dados impuros
    I(9,5) = -9/14 log2(9/14)-(5/14) log2(5/14) = 0,94
> Ganho de informação:
    - Idade: ganho(idade) = I-I_idade = 0,940 - 0,694 = 0,246
    - Renda: ganho(renda) = 0,029
    - Estudante: ganho(estudante) = 0,151
    - Crédito: ganho(crédito) = 0,048
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Carregar o conjunto de dados Iris
iris = load_iris()
X = iris.data
y = iris.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criar o classificador de árvore de decisão
clf = DecisionTreeClassifier()

# Treinar o classificador com os dados de treinamento
clf.fit(X_train, y_train)

# Fazer previsões com os dados de teste
y_pred = clf.predict(X_test)

# Calcular a acurácia das previsões
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia:", accuracy)

# Gerar o gráfico da árvore de decisão
fig = plt.figure(figsize=(10, 8))
_ = tree.plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names.tolist(), filled=True)

# Exibir o gráfico da árvore de decisão
plt.show()