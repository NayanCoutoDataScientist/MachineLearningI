# Técnicas de clusterização e a aprendizagem não supervisionadas
"""
O que é clusterização?
> A Clusterização de Dados ou Análise de Agrupamentos é uma técnica de mineração de dados multivariados que através
de métodos numéricos e a parti somente das informações das variáveis de cada caso, tem por objetivo agrupar
automaticamente por aprendizado não supervisionado os n casos da base de dados em k grupos, geralmente disjuntos
denominados clusters ou agrupamentos.

O que é algoritmo não supervisionado?
> O aprendizado não supervisionado aprende com dados de teste que não foram rotulados, classificados ou categorizados
previamente.
"""

# Pré-processamento dos dados de entrada
"""
O que é pré processamento de dados?
> Pré-processamento é a fase mais importante.
> Documentos são transformados em forma numérica.
> Criação ed BOW (Bag of Words).
> BOW é uma representação numérica da coleção dos documentos.
"""

# Clusterização hierárquica e K-Means
"""
O que é K-Means?
> K-Means é um algoritmo de clusterização (ou agrupamento).
> É um algoritmo de aprendizado não supervisionado (ou seja, que não precisa e inputs de confirmação externos).

Funcionamento
> Atribuição ao cluster: calcula-se a distância entre todos os pontos de dados e cada um dos centroides.
> Movimentação de centroide: uma vez que os pontos foram atribuídos ao cluster conforme distância, há um recálculo dos
valores dos centroides.

Vejamos as principais etapas envolvidas em um algoritmo K-means:

> Definição de um ‘K’, o número de clusters da sua base de dados.
> Definir o centroide para cada cluster para realizar o agrupamento.
> Calcular, para cada ponto, o centroide de menor distância.
> Reposicionar o centroide, devendo ser a média da posição de todos os pontos do cluster.
> Os dois últimos passos são repetidos até chegar no resultado final.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

exemplo = "G:/.shortcut-targets-by-id/1-0eJugp2mhHVgylXYxSrV79OPbp72CNB/Cloud Drive/Documentos/Arquivos PDF, PPT, DOC/Ciências de Dados - Anhanguera Ampli/Inteligência Artificial/Machine Learning I/MachineLearningI/Datasets/Vendas-Arquivo2.csv"

dataset = pd.read_csv(exemplo, sep=";")  # nesse momento devemos fazer a leitura do nosso dataset

X = dataset[['MESCOMPRA', 'VALOR', 'PESOGRAMAS']].values

kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(X)  # aplicação do algoritmo K-Means no conjunto de dados

# Exibir os clusters antes do K-Means
plt.scatter(X[:, 0], X[:, 1], c='blue', label='Dados Originais')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], c='red', marker='x', label='Centroides Iniciais')
plt.xlabel('MESCOMPRA')
plt.ylabel('VALOR')
plt.title('Clusters Antes do K-Means')
plt.legend()
plt.show()

# Executar o algoritmo K-Means
while True:
    X_clustered = kmeans.predict(X)
    old_centroids = kmeans.cluster_centers_.copy()
    kmeans.cluster_centers_ = np.array([X[X_clustered == i].mean(axis=0) for i in range(kmeans.n_clusters)])
    if np.all(old_centroids == kmeans.cluster_centers_):
        break

# Exibir os clusters após o K-Means
plt.scatter(X[:, 0], X[:, 1], c=X_clustered, cmap='viridis', label='Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', label='Centroides Finais')
plt.xlabel('MESCOMPRA')
plt.ylabel('VALOR')
plt.title('Clusters Após o K-Means')
plt.legend()
plt.show()
