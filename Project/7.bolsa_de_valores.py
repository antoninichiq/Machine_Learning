#Usando no ambiente tensorflow_!
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import os

os.environ['CUDA_VISIBLE_DEVICES']='-1' #desativar gpu

dataset = pd.read_csv("C:/Users/anton/OneDrive/Documentos/Curso_ML/Datasets/archive (16)/GSPC.csv")
print(dataset.shape)
print(dataset.head())

#Mostrando o gráfico no período dos 3 primeiros anos (1950-1952)
x = dataset['Date']
y = dataset['Close']
# plt.figure(figsize=(10,5))
# plt.plot(x,y)
# plt.show()

dataset = pd.concat([x,y],axis=1)
dataset.columns=['x','y']
print(dataset.head())

from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
#Normalizando os dados
scaler = MinMaxScaler(feature_range=(0,1))
dataset['y'] = scaler.fit_transform(dataset['y'].values.reshape(-1,1))
tamanho_treino  = int(len(dataset) * 0.67)
treino,teste = dataset[0:tamanho_treino],dataset[tamanho_treino:len(dataset)]
teste = teste.reset_index(drop=True) 

# Criando a função para agrupar os dados em windows
def gera_dataset(dataset, tamanho_janela = 1):
    dataA, dataB = [], []
    for i in range(len(dataset)-tamanho_janela):
        a = dataset.loc[(dataset.index[0]+i):(dataset.index[0]+i+tamanho_janela-1)]['y'].values
        dataA.append(a)
        dataB.append(dataset.loc[[dataset.index[0]+i+tamanho_janela]]['y'].values)
    return np.asarray(dataA, dtype=np.float32), np.asarray(dataB, dtype=np.float32)

#USANDO STATEFULL (o resultado de lote serve como input pro proximo) - relacao temporal
tamanho_janela = 10
x_treino,y_treino = gera_dataset(treino,tamanho_janela=tamanho_janela)
x_teste,y_teste = gera_dataset(teste,tamanho_janela=tamanho_janela)

x_treino = np.reshape(x_treino,(x_treino.shape[0],x_treino.shape[1],1)) 
x_teste = np.reshape(x_teste,(x_teste.shape[0],x_treino.shape[1],1)) 
# print(x_treino.shape)# (340,10,1)
# print(x_teste.shape)#(140,10,1)


modelo = Sequential()
modelo.add(LSTM(4,batch_input_shape=(2,x_treino.shape[1],x_treino.shape[2]),stateful=True))
#Em batch_input_shape nós precisamos informar 3 valores: (batch_size,n_timesteps,n_features)
#Precisamos usar um batch_size que seja divisivel tanto pelo número de amostras do x_treino como x_teste (n_amostras,features,timesteps)

modelo.add(Dense(1)) 
modelo.compile(loss='mean_squared_error',optimizer='adam')

modelo.fit(x_treino,y_treino,epochs=200,batch_size=2,shuffle=False,verbose=2) #importante usar shuffle como false qnd ativar stateful

from sklearn.metrics import mean_squared_error

#Previsoes
previsao_treino = modelo.predict(x_treino,batch_size=2)
previsao_teste = modelo.predict(x_teste,batch_size=2)

# Inverte as previsões por conta da normalização
previsao_treino = scaler.inverse_transform(previsao_treino)
y_treino = scaler.inverse_transform(y_treino)
previsao_teste = scaler.inverse_transform(previsao_teste)
y_teste = scaler.inverse_transform(y_teste)

# Calcula o RMSE
score_treino = math.sqrt(mean_squared_error(y_treino, previsao_treino))
print('Score em Treino: %.2f RMSE' % (score_treino)) 
score_teste = math.sqrt(mean_squared_error(y_teste, previsao_teste))
print('Score em Teste: %.2f RMSE' % (score_treino))

# Mostrando as previsões do modelo:
previsto = teste[tamanho_janela: ].copy()
previsto['y'] = previsao_teste # coloca os valores das previsoes do modelo dentro dessa variável
fig, ax = plt.subplots(1, figsize=(10, 5))
treino['y'] = scaler.inverse_transform(treino['y'].values.reshape(-1,1))
teste['y'] = scaler.inverse_transform(teste['y'].values.reshape(-1,1))
ax.plot(treino['y'], label='treino', linewidth=2)
ax.plot(teste['y'][tamanho_janela:], label='teste', linewidth=2)
ax.plot(previsto['y'], label='previsões', linewidth=2)
ax.set_ylabel('Função', fontsize=14)
ax.legend(loc='best', fontsize=16)

# Olhando mais de perto:

fig, ax = plt.subplots(1, figsize=(10, 5))
ax.plot(teste['y'][tamanho_janela:], label='teste', linewidth=2)
ax.plot(previsto['y'], label='previsões', linewidth=2)
ax.set_ylabel('Função', fontsize=14)
ax.legend(loc='best', fontsize=16)