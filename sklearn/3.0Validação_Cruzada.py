import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict_Ver1.1.csv")

#Excluindo features irrelevantes
arquivo.drop('Serial No.', axis = 1, inplace = True)

#Definindo variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold #kfold para problemas de regressao e StratifiedKFold para problemas de classificação
from sklearn.linear_model import LinearRegression

modelo = LinearRegression()
kfold = KFold(n_splits=5)
resultado = cross_val_score(modelo,x,y,cv=kfold)
print(resultado.mean()) # média da performance de todas as 5 rodadas 