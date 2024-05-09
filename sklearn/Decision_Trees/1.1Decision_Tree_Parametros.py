from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

#print(x.head())

from sklearn.tree import DecisionTreeClassifier #O dataset é um problema de classificação


import numpy as np
from sklearn.model_selection import GridSearchCV

#Definindo os valores que serão testados em DecisionTree
minimos_split = np.array([2,3,4,5,6,7,8]) #cortar as variaveis que tem varios valores, pra decidir um norte pra arvore. Ex B<= 25
maximo_nivel = np.array([3,4,5,6]) #profundidade máxima
algoritimo = ['gini','entropy']
valores_grid = {'min_samples_split': minimos_split, 'max_depth':maximo_nivel,'criterion':algoritimo}

#Criação do modelo
modelo = DecisionTreeClassifier()

#Criando os grids
gridDecisionTree = GridSearchCV(estimator= modelo, param_grid=valores_grid,cv=5)
gridDecisionTree.fit(x,y)

print("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
print("Máxima profundidade: ", gridDecisionTree.best_estimator_.max_depth)
print ("Algoritmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
print ("Acurácia: ", gridDecisionTree.best_score_)

#Visualizar a árvore de decisão
import graphviz
from sklearn.tree import export_graphviz

#Criando o arquivo que irá armazenar a árvore
arquivo = "C:/Users/anton/grafico3.dot"
melhor_modelo = DecisionTreeClassifier(min_samples_split=gridDecisionTree.best_estimator_.min_samples_split,
                                       max_depth=gridDecisionTree.best_estimator_.max_depth,
                                       criterion=gridDecisionTree.best_estimator_.criterion)
melhor_modelo.fit(x,y)

#Gerando o gráfico da árvore de decisão
export_graphviz(melhor_modelo, out_file = arquivo, feature_names = iris.feature_names)
with open(arquivo) as aberto:
    grafico_dot = aberto.read()
h = graphviz.Source(grafico_dot)
h.view()
