import pandas as pd
pd.set_option('display.max_columns', None)
arquivo = pd.read_excel('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (11)/Concrete_Data.xls')
print(arquivo.head())
print(arquivo.dtypes)
print(arquivo.columns)
faltantes = arquivo.isnull().sum()
faltantes_pocentagem = (arquivo.isnull().sum() / len(arquivo['Concrete compressive strength(MPa, megapascals) '])) * 100
print(faltantes_pocentagem)

y = arquivo['Concrete compressive strength(MPa, megapascals) ']
x = arquivo.drop('Concrete compressive strength(MPa, megapascals) ',axis=1)

from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold,cross_val_score
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet

modelo = BaggingRegressor(n_estimators=100,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,x,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

modelo1 = BaggingRegressor(estimator=GradientBoostingRegressor(n_estimators=100),n_estimators=100,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo1,x,y,cv=kfold,n_jobs=-1,scoring='neg_mean_absolute_error') # r2 é padrao para problema de regressao. Mas poderiamos trocar para 'neg_mean_absolute_error'
print(resultado.mean()) # obtendo o erro do dado previsto. Se medir 50 para a variavel target, a resposta pode varias de 46 até 54

modelo = BaggingRegressor(estimator=LinearRegression(),n_estimators=100,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,x,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

modelo = BaggingRegressor(estimator=Ridge(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,x,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

modelo = BaggingRegressor(estimator=Lasso(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,x,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

modelo = BaggingRegressor(estimator=ElasticNet(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
kfold = KFold(n_splits=3,shuffle=True)
resultado = cross_val_score(modelo,x,y,cv=kfold,n_jobs=-1)
print(resultado.mean())
