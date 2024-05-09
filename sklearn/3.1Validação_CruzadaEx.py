import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict_Ver1.1.csv")

#Excluindo features irrelevantes
arquivo.drop('Serial No.', axis = 1, inplace = True)

#Definindo variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression

def validacao_cruzada_kfold(modelo, x, y, kfold):
    result_ridge = cross_val_score(modelo,x,y,cv=kfold)
    return result_ridge.mean()
    
def modelos_regressao(x,y):
    from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
    modelos = [LinearRegression(), Ridge(), Lasso(), ElasticNet()]
    
    performance = []
    kfold = KFold(n_splits=10)
    for i in modelos:
        result = validacao_cruzada_kfold(i, x, y, kfold)
        performance.append(result)
    
    print(f"Regressão Linear: {performance[0]}\nRegressão Ridge: {performance[1]}\nRegressão lasso: {performance[2]}\nRegressão ElasticNet: {performance[3]}")
    
modelos_regressao(x,y)