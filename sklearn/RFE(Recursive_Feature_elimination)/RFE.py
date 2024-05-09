import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

#pd.set_option('display.max_columns',10)
#pd.set_option('display.width',320)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict.csv")
arquivo.drop('Serial No.',axis=1,inplace=True)

y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis = 1)

modelo = Ridge()

rfe = RFE(estimator=modelo,n_features_to_select=5) #escolha as 5 melhores variáveis
melhores = rfe.fit(x,y)

print('Número de atributos:', melhores.n_features_)
print('Atributos selecionados:', melhores.support_)
print('Ranking dos atributos:', melhores.ranking_) #ordem de seleção 
