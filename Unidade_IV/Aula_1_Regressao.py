# Introdução à regressão; modelos de função de base linear e não linear; verossimilhança

import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# carregamento da base de dados
diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# uso de apenas uma feature da base de dados
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Divisão da base de dados em conjuntos de treinamento e teste
diabetes_X_train = diabetes_X[:-20]
diabetes_X_test = diabetes_X[-20:]

# Divisão das saídas em treino e teste
diabetes_y_train = diabetes_y[:-20]
diabetes_y_test = diabetes_y[-20:]

# chama a função de regressão linear
regr = linear_model.LinearRegression()

# treinamento do modelo
regr.fit(diabetes_X_train, diabetes_y_train)

# previsão do modelo com os conjuntos de teste
diabetes_y_pred = regr.predict(diabetes_X_test)

# Coeficiente
print("Coeficiente: \n", regr.coef_)

# Erro médio quadrado
print("Erro médio quadrado: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))

# O coeficiente de previsão: 1 é a previsão perfeita
print("Coeficiente de previsão: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Gráfico das saídas
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)
plt.title("Regressão Linear")
plt.xticks(())
plt.yticks(())
plt.show()

# Regressão Linear Bayesiana Aproximação senoidal
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import BayesianRidge, LinearRegression

# Geração de dados simulados com pesos aleatórios
np.random.seed(0)
n_samples, n_features = 100, 100
X = np.random.randn(n_samples, n_features)

# Criação de pesos com lambda = 4
lambda_ = 4.0
w = np.zeros(n_features)

# Utilizando apenas 10 pesos
relevant_features = np.random.randint(0, n_features, 10)
for i in relevant_features:
    w[i] = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(lambda_))

# Precisão Alpha = 50
alpha_ = 50
noise = stats.norm.rvs(loc=0, scale=1.0 / np.sqrt(alpha_))
size = n_samples

# Configuração para saída
y = np.dot(X, w) + noise

# Ajuste do modelo e compara com o OLS
clf = BayesianRidge(compute_score=True)
clf.fit(X, y)
ols = LinearRegression()
ols.fit(X, y)

# Implementação para geração do gráfico
lw = 2
plt.figure(figsize=(6, 5))
plt.title("Pesos do modelo")
plt.plot(clf.coef_, color="lightgreen", linewidth=lw, label="Regressão Linear Bayesiana")
plt.xlabel("Features")
plt.ylabel("Valores dos Pesos")
plt.legend(loc="best", prop=dict(size=12))
plt.show()

# Regressão não linear com o método Gauss-Newton
import numpy as np
from scipy.optimize import curve_fit
import pylab
import pandas as pd

# Caminho para o arquivo CSV
caminho_arquivo = ('G:/.shortcut-targets-by-id/1-0eJugp2mhHVgylXYxSrV79OPbp72CNB/Cloud Drive/Documentos/'
                   'Arquivos PDF, PPT, DOC/Ciências de Dados - Anhanguera Ampli/Inteligência Artificial/'
                   'Machine Learning I/MachineLearningI/Datasets/Bovinos-Arquivo5.csv')

sh1 = pd.read_csv(caminho_arquivo, sep=';', encoding='latin1')
print(sh1)

xdados = sh1.iloc[:, 3].values / 100
ydados = sh1.iloc[:, 2].values / 100


# Definição do modelo
def func(x, p1, p2):
    return p1 * np.exp(p2 * xdados)


# Chamada de curve_fit
popt, pcov = curve_fit(func, xdados, ydados, p0=(10.0, -1.0))
p1, p2 = popt

# Impressão dos parâmetros ótimos
print("Optimal parameters are p1 = %g, p2=%g" % (p1, p2))

# Plotagem dos resultados
yfitted = func(xdados, *popt)
pylab.plot(xdados, ydados, 'o', label='dados $y_i$')
pylab.plot(xdados, yfitted, '-', label='fit $f(x_i)$')
pylab.xlabel('x')
pylab.ylabel('y')
pylab.legend()
pylab.show()