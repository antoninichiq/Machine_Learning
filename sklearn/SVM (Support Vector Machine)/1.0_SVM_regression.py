import pandas as pd
arquivo = pd.read_excel('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (11)/Concrete_Data.xls')
arquivo.head()

y = arquivo['Concrete compressive strength(MPa, megapascals) ']
x = arquivo.drop('Concrete compressive strength(MPa, megapascals) ',axis=1)

from sklearn.svm import SVR #regressao
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import MinMaxScaler # como o svm trabalha com calculo de distancias, é importante normalizar esses dados

normalizador = MinMaxScaler(feature_range=(0,1))
X_norm = normalizador.fit_transform(x)

modelo = SVR()
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,X_norm,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

import numpy as np
from sklearn.model_selection import GridSearchCV
 # Ver caderno para entender esses parâmetros
c = np.array([1.0,0.95,1.05,1.1,1.2,2.0,0.9,0.8]) # valor que multiplica os erros na eq. de Lagrange
kernel = ['linear','poly','rbf','sigmoid']
polinomio = np.array([2,3,4])
epsilon = np.array([0.1,0.2,0.05])
valores_grid = {'C':c,'kernel':kernel,'degree':polinomio,'epsilon':epsilon}

modelo = SVR()

kfold = KFold(n_splits=3,shuffle=True)
gridSVR = GridSearchCV(estimator=modelo, param_grid=valores_grid,cv=kfold,n_jobs=-1)
gridSVR.fit(X_norm,y)

print("Melhor valor constante: ", gridSVR.best_estimator_.C)
print("Melhor kernel: ", gridSVR.best_estimator_.kernel)
print("Melhor grau polinômio: ", gridSVR.best_estimator_.degree)
print("Melhor epsilon: ", gridSVR.best_estimator_.epsilon)
print('R2: ',gridSVR.best_score_)

#Fazendo uma nova rodada, dado os valores passados
c = np.array([2,4,8,16,32]) # valor que multiplica os erros na eq. de Lagrange
kernel = ['linear','poly','rbf','sigmoid']
polinomio = np.array([2,3,4])
epsilon = np.array([0.1,0.2,0.05])
valores_grid = {'C':c,'kernel':kernel,'degree':polinomio,'epsilon':epsilon}

modelo = SVR()

kfold = KFold(n_splits=3,shuffle=True)
gridSVR = GridSearchCV(estimator=modelo, param_grid=valores_grid,cv=kfold,n_jobs=-1)
gridSVR.fit(X_norm,y)

print("Melhor valor constante: ", gridSVR.best_estimator_.C)
print("Melhor kernel: ", gridSVR.best_estimator_.kernel)
print("Melhor grau polinômio: ", gridSVR.best_estimator_.degree)
print("Melhor epsilon: ", gridSVR.best_estimator_.epsilon)
print('R2: ',gridSVR.best_score_)



