from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

# print(x.head())
# print(x.shape)
# print(iris)
# print(x.dtypes)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

normalizador = MinMaxScaler(feature_range=(0,1))
X_norm = normalizador.fit_transform(x)

valores_K = np.array([3,4,7,9,11])
calculo_distancia = ['minkowski','chebyshev']
valores_p = np.array([1,2,3,4])
valores_grid = {'n_neighbors':valores_K, 'metric': calculo_distancia,'p':valores_p}

modelo = KNeighborsClassifier()

gridKNN = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv= 5)
gridKNN.fit(X_norm,y)

print ("Melhor acurácia: ", gridKNN.best_score_)
print ("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
print ("Método distância: ", gridKNN.best_estimator_.metric)
print ("Melhor valor p: ", gridKNN.best_estimator_.p)