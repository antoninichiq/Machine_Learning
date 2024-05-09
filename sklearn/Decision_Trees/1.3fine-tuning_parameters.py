import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict_Ver1.1.csv")

#Excluindo features irrelevantes
arquivo.drop('Serial No.', axis = 1, inplace = True)

#Definindo variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

#Definindo os valores que serão testados em DecisionTree
minimos_split = np.array([2,3,4,5,6,7]) #cortar as variaveis que tem varios valores, pra decidir um norte pra arvore. Ex B<= 25
maximo_nivel = np.array([3,4,5,6,7,9,11]) #profundidade máxima
algoritimo = ['mse','friedman_mse', 'mae']
valores_grid = {'min_samples_split': minimos_split, 'max_depth':maximo_nivel,'criterion':algoritimo}

#Criação do modelo
modelo = DecisionTreeRegressor()

#Criando os grids
gridDecisionTree = GridSearchCV(estimator= modelo, param_grid=valores_grid,cv=5)
gridDecisionTree.fit(x,y)

print("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
print("Máxima profundidade: ", gridDecisionTree.best_estimator_.max_depth)
print ("Algoritmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
print ("Coeficiente R2: ", gridDecisionTree.best_score_)
