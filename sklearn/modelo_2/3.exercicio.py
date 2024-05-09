import pandas as pd
pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (9)/student-mat.csv")
print(arquivo.head())
print(arquivo.dtypes)
for k,v in arquivo.items():
    if arquivo[k].dtypes == 'object' and len(arquivo[k].value_counts()) == 2:
        mapping = {arquivo[k].unique()[0]: 0, arquivo[k].unique()[1]: 1}
        arquivo[k] = arquivo[k].map(mapping)

#print(arquivo.dtypes)

# print(arquivo.head())
#print(concatenado['guardian'].value_counts())
one_hot_encode = ['Mjob','Fjob','reason','guardian']
for i in one_hot_encode:
    encode = pd.get_dummies(arquivo[i])
    arquivo.drop(i, axis = 1, inplace=True)
    concatenado = pd.concat([arquivo, encode],axis=1)
print(arquivo.head())
print(arquivo.dtypes)

faltantes = concatenado.isnull().sum()
faltantes_percentual = (concatenado.isnull().sum() / len(concatenado['G3'])) * 100
print(faltantes_percentual)

y = concatenado['G3']
x = concatenado.drop('G3',axis=1)

from sklearn.ensemble import AdaBoostRegressor
from sklearn.model_selection import KFold, cross_val_score

modelo = AdaBoostRegressor()
kfold = KFold(n_splits=2)
resultado = cross_val_score(modelo,x,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

#Melhores parametros
import numpy as np
from sklearn.model_selection import GridSearchCV

valores_grid = {'learning_rate':np.array([0.3,0.2,0.1,0.05])}

modelo = AdaBoostRegressor(n_estimators=500)

gridAdaBoost = GridSearchCV(estimator=modelo, param_grid=valores_grid,cv=5,n_jobs=-1)
gridAdaBoost.fit(x,y)

print("Melhor taxa de aprendizagem: ", gridAdaBoost.best_estimator_.learning_rate)
print("Acurácia: ",gridAdaBoost.best_score_)



