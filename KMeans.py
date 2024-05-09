from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('display.max_columns',30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data, columns=[dados.feature_names])
y = pd.Series((dados.target))

#Normalização
from sklearn.preprocessing import MinMaxScaler
normalizador = MinMaxScaler(feature_range= (0,1))
x_norm = normalizador.fit_transform(x)

from sklearn.cluster import KMeans
modelo = KMeans(n_clusters=2,random_state=16,n_init=10)
modelo.fit(x_norm)

#print(modelo.cluster_centers_)
#print(modelo.predict(x_norm))

clusters = modelo.predict(x_norm)
def compara(resultado1,resultado2):
    acertos = 0
    for i in range(len(resultado1)):
        if resultado1[i] == resultado2[i]:
            acertos += 1
        else:
            pass
    return acertos/len(resultado1)

resultado = compara(clusters,y)
print(resultado)
        
#calculando a acurácia com funções prontas do sklearn
from sklearn.metrics import accuracy_score
print(accuracy_score(y,clusters))
