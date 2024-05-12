from sklearn.datasets import fetch_california_housing
import pandas as pd
pd.set_option('display.max_columns',None)
housing = fetch_california_housing()
x = pd.DataFrame(housing.data,columns=[housing.feature_names])
y = pd.Series(housing.target)

print(x.shape)

from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste=train_test_split(x,y,test_size=0.3)

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout

modelo = Sequential()
modelo.add(Dense(30,input_dim=8,kernel_initializer='normal',activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(30,kernel_initializer='normal',activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(1,kernel_initializer='normal',activation='linear'))

from keras.optimizers import Adam
otimizador = Adam()

modelo.compile(loss='mean_squared_error',optimizer=otimizador,metrics=['mean_absolute_error'])
historico = modelo.fit(x_treino,y_treino,epochs=100,batch_size=200,validation_data=(x_teste,y_teste),verbose=1)

import matplotlib.pyplot as plt
mae_treino = historico.history['mean_absolute_error']
mae_teste = historico.history['val_mean_absolute_error']

epochs = range(1,len(mae_treino)+1)

plt.plot(epochs,mae_treino,'-g',label="MAE Dados de Treino")
plt.plot(epochs,mae_teste,'-b',label="MAE Dados de Teste")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel("MAE")
plt.show() 
