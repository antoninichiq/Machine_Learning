# Não consegui baixar o dataset
# É um dataset para verificar se o cogumelo é comestível (EDIBLE) ou não (POISONOUS)
# Esse dataset é todo em string, então precisamos binarizar com 0 ou 1 ou fazer o one hot encoding

import pandas as pd
pd.set_option('diplay.max_columns',89)
arquivo = pd.read_csv("Caminho/arquivo.csv")

# Verificando as variáveis de cada coluna. 
print([arquivo[c].value_counts() for c in list(arquivo.columns)]) # A maioria tem mais de 2 strings, então faremos o one hot encoding

y = arquivo['mushroom']
x = arquivo.drop('mushromm',axis=1)

x_encoding = pd.get_dummies(x)
x_encoding.head()

pd.set_option('display.max_rows',None)
faltantes = x_encoding.isnull().sum()
faltantes_percentual = (x_encoding.isnull().sum() / len(y)) * 100
print(faltantes_percentual)

y = y.replace('EDIBLE', 0)
y = y.replace('POISONOUS',1)
y.value_counts

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

modelo = ExtraTreesClassifier(n_estimators=50,n_jobs=-1)
skfold = StratifiedKFold(n_splits=3,shuffle=True) # shuffle true embaralha os dados. Uma boa prática é sempre colocar como True. Às vezes não faz diferença
resultado = cross_val_score(modelo,x,y,cv=skfold,n_jobs=-1)
print(resultado.mean())

# resultado.mean() = 1.0