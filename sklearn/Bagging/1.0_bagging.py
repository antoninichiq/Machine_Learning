import pandas as pd
pd.set_option('display.max_columns',None)
arquivo = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (6)/recipeData.csv', encoding='ISO-8859-1')

selecao = arquivo.loc[arquivo['StyleID'].isin([7,10,134,9,4,30,86,12,92,6,175,39])]
selecao.drop(['BeerID', 'PrimingAmount' ,'PrimingMethod', 'UserId', 'Style', 'URL', 'Name'],axis=1,inplace=True)
print(selecao.head())

selecao['SugarScale'] = selecao['SugarScale'].replace('Specific Gravity', 0)
selecao['SugarScale'] = selecao['SugarScale'].replace('Plato', 1)

#Transformando variáveis texto na coluna 'BrewMethod' em categorias com one hot encoding
brewmethod_encode = pd.get_dummies(selecao['BrewMethod'])

#Excluindo a coluna de texto 'BrewMethod'
selecao.drop('BrewMethod', axis = 1, inplace = True)

#Inserindo as variáveisIone hot encode novamente no dataset
concatenado = pd.concat([selecao, brewmethod_encode], axis=1)

concatenado['PitchRate' ].fillna(concatenado['PitchRate' ].mean(), inplace=True)
concatenado. fillna(concatenado.median(), inplace=True)

y = concatenado['StyleID']
x = concatenado.drop('StyleID', axis=1)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

# modelo = DecisionTreeClassifier()
# skfold = StratifiedKFold(n_splits=3)
# resultado = cross_val_score(modelo,x,y,cv=skfold,n_jobs=-1)
# print(resultado.mean())

# import numpy as np
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import GridSearchCV

# minimos_split = np.array([2, 3, 4, 5, 6, 7, 8])
# maximo_nivel = np.array([5, 6, 7, 8, 9, 10, 11])
# minimo_leaf = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# valores_grid = {'min_samples_split': minimos_split, 'min_samples_leaf': minimo_leaf, 'max_depth': maximo_nivel}

# #Criação do modelo:
# modelo = DecisionTreeClassifier()

# #Criando os grids:
# gridDecisionTree = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=3, n_jobs =- 1)
# gridDecisionTree.fit(x,y)

# #Imprimindo os melhores parâmetros:
# print ("Minimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
# print ("Maxima profundidade: ", gridDecisionTree.best_estimator_.max_depth)
# print ("Minimo leaf: ", gridDecisionTree.best_estimator_.min_samples_leaf)
# print ("Acuracia: ", gridDecisionTree.best_score_)

from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

modelo = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo,x,y,cv=skfold,n_jobs=-1)
print(resultado.mean())

# Outros exemplos
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score

modelo = BaggingClassifier(estimator=LogisticRegression(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo,x,y,cv=skfold,n_jobs=-1)
print(resultado.mean())

from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

modelo = BaggingClassifier(estimator=GaussianNB(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo,x,y,cv=skfold,n_jobs=-1)
print(resultado.mean())
