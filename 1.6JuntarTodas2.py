import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict_Ver1.1.csv")
# print(arquivo.head())
# print(arquivo.shape)
# print(arquivo.dtypes)
# faltantes = arquivo.isnull().sum()
# print(faltantes)

#Excluindo features irrelevantes
arquivo.drop('Serial No.', axis = 1, inplace = True)

#Definindo variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30, random_state=14)

def modelos_regressao(x_treino,x_teste,y_treino,y_teste):
    from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
    reg = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elastic = ElasticNet()
    
    reg.fit(x_treino,y_treino)
    ridge.fit(x_treino,y_treino)
    lasso.fit(x_treino,y_treino)
    elastic.fit(x_treino,y_treino)
    
    result_reg = reg.score(x_teste,y_teste)
    result_ridge = ridge.score(x_teste,y_teste)
    result_lasso = lasso.score(x_teste,y_teste)
    result_elastic = elastic.score(x_teste,y_teste)

    print(f"Regressão Linear: {result_reg}\nRegressão Ridge: {result_ridge}\nRegressão lasso: {result_lasso}\nRegressão ElasticNet: {result_elastic}")
    
modelos_regressao(x_treino,x_teste,y_treino,y_teste)