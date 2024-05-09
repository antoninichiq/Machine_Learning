import pandas as pd
# importando com url, poderia baixar: archive (12)
arquivo = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data',
names=['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])

arquivo2 = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/balance-scale/balance-scale.data')
print(arquivo.head())
print(arquivo.dtypes)

faltantes = arquivo.isnull().sum()
faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['Class'])) * 100
print(faltantes_percentual)
print(arquivo['Class'].value_counts())

y = arquivo['Class']
x = arquivo.drop('Class',axis=1)

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.preprocessing import MinMaxScaler # como o svm trabalha com calculo de distancias, é importante normalizar esses dados

normalizador = MinMaxScaler(feature_range=(0,1))
X_norm = normalizador.fit_transform(x)

modelo = SVC()
kfold = StratifiedKFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,X_norm,y,cv=kfold,n_jobs=-1)
print(resultado.mean())


import numpy as np
from sklearn.model_selection import GridSearchCV
 # Ver caderno para entender esses parâmetros
c = np.array([1.0,0.95,1.05,1.1,1.2,2.0,0.9,0.8]) # valor que multiplica os erros na eq. de Lagrange
kernel = ['linear','poly','rbf','sigmoid']
polinomio = np.array([2,3,4])
gamma = ['auto','scale']
valores_grid = {'C':c,'kernel':kernel,'degree':polinomio,'gamma':gamma}

modelo = SVC()

kfold = StratifiedKFold(n_splits=3,shuffle=True)
gridSVR = GridSearchCV(estimator=modelo, param_grid=valores_grid,cv=kfold,n_jobs=-1)
gridSVR.fit(X_norm,y)

print("Melhor valor constante: ", gridSVR.best_estimator_.C)
print("Melhor kernel: ", gridSVR.best_estimator_.kernel)
print("Melhor grau polinômio: ", gridSVR.best_estimator_.degree)
print("Melhor epsilon: ", gridSVR.best_estimator_.epsilon)
print('R2: ', gridSVR.best_score_)