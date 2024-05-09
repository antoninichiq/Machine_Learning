from sklearn.datasets import load_iris
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

iris = load_iris()

x = pd.DataFrame(iris.data,columns=[iris.feature_names])
y = pd.Series(iris.target)

#print(x.head())

algoritmo = SelectKBest(score_func=chi2,k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

print("Score: ", algoritmo.scores_)
print("Resultado da transformação:\n", dados_das_melhores_preditoras)
