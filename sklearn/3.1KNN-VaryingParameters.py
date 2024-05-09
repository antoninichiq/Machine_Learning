from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('display.max_columns', 30)
data = load_breast_cancer()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler # função de normalização. Cada coluna tem uma característica diferente. Uma tem ordem de grandeza na ordem de 0.02 e outra na ordem das dezenas. 
from sklearn.model_selection import GridSearchCV

#Normalizando as variáveis preditoras
normalizador = MinMaxScaler(feature_range= (0, 1))
X_norm = normalizador.fit_transform(x)

#Definindo os valores que serão testados no KNN:
valores_K = np.array([3,5,7,9,11])
calculo_distancia = ['minkowski','chebyshev']
valores_p = np.array([1,2,3,4])
valores_grid = {'n_neighbors':valores_K, 'metric': calculo_distancia,'p':valores_p}


#Criação do modelo:
modelo = KNeighborsClassifier()

#Criando os grids:
gridKNN = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv = 5)
gridKNN.fit(X_norm,y)


#Imprimindo os melhores parâmetros:
print ("Melhor acurácia: ", gridKNN.best_score_)
print ("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
print ("Método distância: ", gridKNN.best_estimator_.metric)
print ("Melhor valor p: ", gridKNN.best_estimator_.p)
