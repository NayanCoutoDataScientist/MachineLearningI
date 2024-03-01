# Introdução ao Support Vector Machine (SVM)
"""
SVM
> Support Vector Machine (SVM), é um algoritmo supervisionado usado em machine learning para regressão, classificação
e detecção de anomalias.

Vantagens:
> Eficaz em espaços de alta dimensão.
> Ainda eficaz nos casos em que o número de dimensões é maior que o número de amostras.
> Usa um subconjunto de pontos de treinamento na função de decisão (chamados de vetores de suporte), portanto, também
é eficiente em termos de memória.
> Versátil: diferentes funções do Kernel podem ser especificadas para função de decisão.

Desvantagens:
> Se o número de recursos for muito maior que o número de amostras, evitar sobreajuste (overfitting) na escolha das
funções do Kernel e o termo de regularização é crucial.
> Os SVM não fornecem estimativas de probabilidade diretamente. Elas são calculadas usando uma validação cruzada de
5-fold (o que pode ser custoso)
"""

# SVM para problemas lineares
#%%
# Bibliotecas e módulos necessários
import numpy as np
from sklearn.datasets import load_iris
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

#%%
#Carrega base de dados iris
iris = load_iris()

#%%
# seleciona coluna das larguras e comprimento de pétalas
dados = iris['data'][0:100, (2,3)]
iris

#%%
# Seleciona a classe => setosa
setosa = (iris['target'][0:100]==0).astype(np.float64)

#%%
# Criando o modelo linear
svm_clf = Pipeline([
    ('scaler', StandardScaler()),
    ('linear_svc', LinearSVC(C=1, loss='hinge'))
])
svm_clf.fit(dados, setosa)

#%%
# Usando o modelo
nova_flor_A = [2, 0.3] # 2cm * 0,3cm
nova_flor_B = [4, 1]

#%%
# Realiza a predição
predicao_A = svm_clf.predict([nova_flor_A])
predicao_B = svm_clf.predict([nova_flor_B])

#%%
# Formata uma mensagem
def resultado(predicao, flor):
    if predicao == 1:
        flor_type = 'setosa'
    else:
        flor_type = 'versicolor'
    print(f'A flor {flor} é {flor_type}.')

#%%
resultado(predicao_A, "A")
resultado(predicao_B, "B")

#%%
# SVM para problemas não-lineares
# Bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets


def criar_grade(x, y, h=0.02):
    # Cria uma grade para plotagem
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    x2, y2 = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )
    return x2, y2


def plotar_limites(ax, classificador, x2, y2, **parametros):
    # Colore os limites de cada classe
    Z = classificador.predict(np.c_[x2.ravel(), y2.ravel()])
    Z = Z.reshape(x2.shape)
    saida = ax.contourf(x2, y2, Z, **parametros)
    return saida


#%%
# Carrega a base de dados iris
iris = datasets.load_iris()

#%%
# Seleciona coluna das largura e comprimento de pétalas
X = iris.data[:, 2:4]
y = iris.target

#%%
# Instancia o SVM e ajusta os dados
C = 1.0
modelos = (
    svm.SVC(kernel='linear', C=C),
    svm.LinearSVC(C=C, max_iter=10000),
    svm.SVC(kernel='rbf', gamma=0.7, C=C),
    svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)
)
modelos = (classificador.fit(X, y) for classificador in modelos)

#%%
# Títulos dos gráficos
titulos = (
    'SVC com kernel linear',
    'Linear SVC',
    'SVC com kernel rbf',
    'SVC com kernel polinomial grau 3'
)

#%%
# Configurar uma grade 2x2 para plotagem
fig, sub = plt.subplots(2, 2)
plt.subplots_adjust(wspace=0.45, hspace=0.45)
X0, X1 = X[:, 0], X[:, 1]
x2, y2 = criar_grade(X0, X1)

#%%
# para cada modelo, gere um gráfico
for classificador, titulo, ax in zip(modelos, titulos, sub.flatten()):
    plotar_limites(ax, classificador, x2, y2, cmap=plt.cm.viridis, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.viridis, s=20, edgecolors='k')
    ax.set_xlim(x2.min(), x2.max())
    ax.set_ylim(y2.min(), y2.max())
    ax.set_xlabel('Comprimento da Pétala')
    ax.set_ylabel('Largura da Pétala')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(titulo)

#%%
#plt.savefig("comparativo.png", dpi=300)
plt.show()