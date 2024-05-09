from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

iris = load_iris()

x = pd.DataFrame(iris.data,columns=[iris.feature_names])
y = pd.Series(iris.target)

#print(x.head())

#Selecionando duas variáveis com o maior F-Value
algoritmo = SelectKBest(score_func=f_classif,k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

print("Score: ", algoritmo.scores_)
print("Resultado da transformação:\n", dados_das_melhores_preditoras )
