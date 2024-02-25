# Classificadores elementares (Template Matching e classificadores baseados em distância mínima)
"""
O que é Template Matching?
> Template Matching é uma técnica de processamento de uma imagem digital para encontrar pequenas partes de uma imagem
que correspondam a uma imagem completa. Pode ser usado na manufatura como parte do controle de qualidade, uma forma
de navegar um robô móvel, ou como uma forma de detectar bordas em imagens.

Os principais desafios na tarefa de correspondência de modelos são: oclusão, detecção de transformações não rígidas,
iluminação e alterações de fundo, desordem de fundo e mudanças de escala.

O que é classificação?
> Processo de extração de informações em imagens para o reconhecimento de padrões e objetos homogêneos.
> Associa cada píxel de uma imagem a um "rótulo" descrevendo o objeto real.

Classificadores
> Píxel a Píxel: usam de forma individual a informação espectral de cada píxel na busca por regiões homogêneas.
> Regiões: utilizam a informação espectral de cada píxel e a relação espacial da vizinhança.

Distância minima
> Examina as distâncias entre um píxel e as médias das classes e atribui esse píxel a classe que apresentar a menor
distância.
"""

# Métricas para avaliação de modelos (matriz de confusão, acurácia, precisão, recall, especificidade e análise da curva Receiver Operating Characteristic Curve (ROC))
"""
Acurácia
> A acurácia é a proximidade de um resultado com o seu valor de referência real. Dessa forma, quanto maior o nível de
acuracidade, mais próximo da referência ou valor real é o resultado encontrado.
> Quando falamos especificamente de validação de identidade, acurácia se refere a quão próximos da realidade são os
resultados encontrados de forma automatizada ou com soluções de IA.
    Acurácia = (Verdadeiros Positivos + Verdadeiros Negativos) / (Verdadeiros Positivos + Falsos Positivos + Verdadeiros Negativos + Falsos Negativos)

Precisão
> A precisão é a proporção de exemplos positivos corretamente classificados em relação ao total de exemplos
classificados como positivos. Em outras palavras, a precisão mede a capacidade do modelo de evitar classificar
erroneamente exemplos negativos como positivos. A fórmula da precisão é:
    Precisão = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Positivos)

Recall
> Por outro lado, o recall é a proporção de exemplos positivos corretamente classificados em relação ao total
de exemplos positivos reais. O recall mede a capacidade do modelo de identificar corretamente exemplos positivos.
A fórmula do recall é:
    Recall = Verdadeiros Positivos / (Verdadeiros Positivos + Falsos Negativos)

Matriz de Confusão
> Em análise preditiva, a matriz de confusão (às vezes chamada de 'matriz de erro' ou 'tabela de confusão') é uma
tabela com duas linhas e duas colunas que relata o número de falsos positivos, falsos negativos, verdadeiros positivos e
verdadeiros negativos. Isso permite uma análise mais detalhada do que a mera proporção de classificações corretas
(precisão).
"""

# Algoritmo K-nearest neighbors - KNN
"""As etapas do algoritmo são (ALPAYDIN, 2020):
1 - Primeiro há o recebimento de um dado que não foi classificado previamente;
2 - Há a medição da distância desse novo dado que não foi classificado anteriormente, levando em consideração os dados
que já obtiveram sua classificação.
3 - Obtém X menores distâncias.
4 - Há uma verificação para identificar a classe de cada um dos dados que obtiverem a menor distância e realizar uma
contagem de cada classe.
5 - Verifica a classe que mais apareceu, considerando os dados que obtiverem as menores distâncias.
Essa classe será seu resultado.
6 - Realiza a classificação do dado com a classe que obtemos como resultado na fase 5.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier

# Gerar dados de exemplo
X, y = make_blobs(n_samples=100, centers=2, random_state=42)

# Criar o classificador KNN
k = 3  # número de vizinhos mais próximos a considerar
knn = KNeighborsClassifier(n_neighbors=k)

# Treinar o classificador com os dados existentes
knn.fit(X, y)

# Gerar um novo dado não classificado
new_data = np.array([[0, 0]])

# Medir a distância do novo dado para os dados existentes
distances, indices = knn.kneighbors(new_data)

# Obter as classes dos vizinhos mais próximos
nearest_labels = y[indices]

# Contar a ocorrência de cada classe
unique_classes, class_counts = np.unique(nearest_labels, return_counts=True)

# Encontrar a classe mais frequente
most_frequent_class = unique_classes[np.argmax(class_counts)]

# Classificar o novo dado com a classe mais frequente
new_data_class = most_frequent_class

# Plotar os dados existentes
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', label='Dados existentes')

# Plotar o novo dado não classificado
plt.scatter(new_data[:, 0], new_data[:, 1], c='red', marker='x', label='Novo dado')

# Plotar os vizinhos mais próximos
plt.scatter(X[indices, 0], X[indices, 1], c=nearest_labels.flatten(), cmap='viridis', marker='s', label='Vizinhos mais próximos')

plt.xlabel('Característica 1')
plt.ylabel('Característica 2')
plt.title('Classificação com KNN')
plt.legend()
plt.show()

print(f'O novo dado foi classificado como: {new_data_class}')

"""
Algoritmo K-nearest neighbors (KNN)
> KNN é um dos muitos algoritmos de aprendizagem supervisionada usados no campo de data mining e machine learning,
ele é um classificador onde o aprendizado é baseado no quão similar é um dado (ou vetor) ao outro. O treinamento é
formado por vetores de n dimensões.

Etapas:
1. Calcular distância.
2. Encontrar os pontos vizinhos mais próximos.
3. Selecionar o label para o ponto a ser previsto.
"""

# Estudo e Caso
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd

SEED = 30

exemplo = "G:/.shortcut-targets-by-id/1-0eJugp2mhHVgylXYxSrV79OPbp72CNB/Cloud Drive/Documentos/Arquivos PDF, PPT, DOC/Ciências de Dados - Anhanguera Ampli/Inteligência Artificial/Machine Learning I/MachineLearningI/Datasets/Vendas-Arquivo2.csv"

dataset = pd.read_csv(exemplo, sep=";")  # nesse momento devemos fazer a leitura do nosso dataset

# Definir os dados de treinamento e teste
X_train = dataset[['MESCOMPRA']].values
y_train = dataset[['VALOR']].values
X_test = dataset[['MESCOMPRA']].values
y_test = dataset[['VALOR']].values

np.random.seed(SEED)

model = DecisionTreeClassifier(max_depth=3)

model.fit(X_train, y_train)

predict = model.predict(X_test)

accuracy = accuracy_score(y_test, predict) * 100

print("A acuracia foi de {:.2f}%.".format(accuracy))