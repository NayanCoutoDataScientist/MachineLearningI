# Least-squares para classificação
'Classificação Naive Bayes'

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import urllib.request
from PIL import Image

# Carregamento do dataset
dataset = pd.read_csv('https://raw.githubusercontent.com/ApoorvRusia/SVM-classification-on-Iris-dataset/master/iris.csv', sep=',')
print(dataset)

#visualizar os tipos de Iris

'''img = mpimg.imread('iris_types.jpg')
plt.figure(figsize=(20,40))
plt.axis('off')
plt.imshow(img)'''

url = 'https://raw.githubusercontent.com/ApoorvRusia/SVM-classification-on-Iris-dataset/master/iris_types.jpg'

# Abrir a imagem da URL usando o Pillow
image = Image.open(urllib.request.urlopen(url))

# Converter a imagem para um array numpy
img_array = np.array(image)

# Exibir a imagem
plt.figure(figsize=(20, 40))
plt.axis('off')
plt.imshow(img_array)
plt.show()

# Separando as variáveis preditora e classes
X = dataset.iloc[:, :4]. values
y = dataset['species'].values

# separando dados de teste e treinamento
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=82)

# Importação do modelo SVM
from sklearn.svm import SVC
from sklearn import metrics

# SVC com kernel linear
linear_svc = SVC(kernel='linear').fit(X_train, y_train)
prediction = linear_svc.predict(X_test)
print('Acurácia SVC:', "{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# SVC com kernel polinominal
poly_svc = SVC(kernel = 'poly', degree = 4).fit(X_train, y_train)
prediction = poly_svc.predict(X_test)
print('Acurácia SVC polinomial:', "{:.3f}".format(metrics.accuracy_score(prediction,y_test)))

# Importação do modelo naive bayes
from sklearn.naive_bayes import GaussianNB
nvclassifier = GaussianNB()
nvclassifier.fit(X_train, y_train)

# Prevendo resultados conjunto teste
y_pred = nvclassifier.predict(X_test)
print(f'y pred: {y_pred}')

# Comparando o valor real e previsto
y_compare = np.vstack((y_test, y_pred)).T
print(f'y compare: {y_compare[:5,:]}')

# Calculando o desemprenho do modelo
from sklearn.metrics import confusion_matrix

# Calcular a matriz de confusão
cm = confusion_matrix(y_test, y_pred)

# Calcular as classificações corretas e incorretas metodo 1
a = cm.shape
corrPred = 0
falsePred = 0
for row in range(a[0]):
    for c in range(a[1]):
        if row == c:
            corrPred += cm[row,c]
        else:
            falsePred += cm[row,c]

print('Classificações Corretas: ', corrPred)
print('Classificações Erradas: ', falsePred)
print(f'\n\nAcurácia do modelo Naive Bayes: {corrPred/(cm.sum()): .2f}')

# Calcular as classificações corretas e incorretas metodo 2
corrPred = cm.diagonal().sum()
falsePred = cm.sum() - corrPred

print('Classificações Corretas:', corrPred)
print('Classificações Incorretas:', falsePred)
print(f'\n\nAcurácia do modelo Naive Bayes: {corrPred/(cm.sum()): .2f}')