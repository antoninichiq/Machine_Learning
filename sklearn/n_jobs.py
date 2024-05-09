import pandas as pd
pd.set_option('display.max_columns', 21)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (3)/Admission_Predict_Ver1.1.csv")

#Excluindo features irrelevantes
arquivo.drop('Serial No.', axis = 1, inplace = True)

#Definindo variáveis preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis=1)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

valores = {'alpha':[0.1,0.5,1,2,5,10,25,50,100,150,200,300,500,750,1000,1500,2000,3000,5000],'l1_ratio':[0.02,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]}

modelo = ElasticNet()
procura =GridSearchCV(estimator=modelo, param_grid=valores,cv=5,n_jobs=-1) #n_jobs = -1 manda ele usar todos os núcleos menos 1 da cpu ao mesmo tempo
procura.fit(x,y)

print('Melhor score:', procura.best_score_)
print('Melhor alpha:', procura.best_estimator_.alpha)
print('Melhor l1_ratio:', procura.best_estimator_.l1_ratio)