import pandas as pd
pd.set_option('display.max_columns', 100)
arquivo = pd.read_csv("C:/Users/anton/Downloads/column_2C_weka.csv")
# print(arquivo.head())
# print(arquivo.shape)
# print(arquivo.dtypes)
# faltantes = arquivo.isnull().sum()
# faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['pelvic_inkidence'])) * 100
# print(faltantes_percentual)

arquivo['class'] = arquivo['class'].replace('Abnormal', 1)
arquivo['class'] = arquivo['class']. replace('Normai', 0)

y = arquivo['class']
x = arquivo.drop('class', axis=1)

from sklearn.tree import DecisionTreeClassifier
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

from sklearn.model_selection import cross_val_score, StratifiedKFold

skfold = StratifiedKFold(n_splits=5)
melhor_modelo = DecisionTreeClassifier(min_samples_split=gridDecisionTree.best_estimator_.min_samples_split,
                                       max_depth=gridDecisionTree.best_estimator_.max_depth,
                                       criterion=gridDecisionTree.best_estimator_.criterion)
melhor_modelo.fit(x,y)
resultado = cross_val_score(melhor_modelo,x,y,cv=skfold)

print(resultado.mean())

import graphviz
from sklearn.tree import export_graphviz
#Gerando o gráfico da árvore de decisão
arquivo1 = "C:/Users/anton/grafico2.dot"
export_graphviz(melhor_modelo, out_file = arquivo1, feature_names = ["pelvic_incidence","pelvic_tilt numeric","lumbar_lordosis_angle","sacral_slope","pelvic_radius","degree_spondylolisthesis"])
with open(arquivo1) as aberto:
    grafico_dot = aberto.read()
h = graphviz.Source(grafico_dot)
h.view()



