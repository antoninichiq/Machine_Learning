from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)
print(x.shape)
print(y.shape)
#print(x.head())

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier #O dataset é um problema de classificação

#Separando os dados em folds
skfold = StratifiedKFold(n_splits=5)

#Criação do modelo
modelo = DecisionTreeClassifier()
resultado = cross_val_score(modelo,x,y,cv=skfold)

#Acurácia
print(resultado.mean())


