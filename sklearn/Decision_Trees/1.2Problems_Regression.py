import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict_Ver1.1.csv")

#Excluindo features irrelevantes
arquivo.drop('Serial No.', axis = 1, inplace = True)

#Definindo variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

kfold = KFold(n_splits=5, random_state=8, shuffle=True)

modelo = DecisionTreeRegressor()
resultado = cross_val_score(modelo,x,y,cv=kfold)

print("Coeficiente de determinação R2:", resultado.mean())
