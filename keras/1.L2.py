import warnings
warnings.filterwarnings('ignore')
import keras
from keras.datasets import mnist

(x_treino,y_treino), (x_teste,y_teste) = mnist.load_data()

from keras.utils import to_categorical
y_treino_convertido = to_categorical(y_treino)
y_teste_convertido = to_categorical(y_teste)

from keras.models import Sequential
from keras.layers import Dense,Activation,Dropout
from keras import regularizers

x_treino_remodelado = x_treino.reshape((60000,784))
x_teste_remodelado = x_teste.reshape((10000,784))

x_treino_normalizado = x_treino_remodelado.astype('float32') / 255
x_teste_normalizado = x_teste_remodelado.astype('float32') / 255

modelo = Sequential()
modelo.add(Dense(30, input_dim=784,kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
modelo.add(Dense(30,kernel_initializer='normal',activation='relu',kernel_regularizer=regularizers.l2(0.01)))
modelo.add(Dense(10,kernel_initializer='normal',activation='softmax'))

from keras.optimizers import Nadam
otimizador = Nadam ()

modelo.compile(loss='categorical_crossentropy',optimizer=otimizador,metrics=['acc'])
historico = modelo.fit(x_treino_normalizado,y_treino_convertido,epochs=10,batch_size=200,validation_data=(x_teste_normalizado,y_teste_convertido),verbose=1)

import matplotlib.pyplot as plt
acuracia_treino = historico.history['acc']
acuracia_teste = historico.history['val_acc']

epochs = range(1,len(acuracia_treino)+1)

plt.plot(epochs,acuracia_treino,'-g',label="Acurácia Dados de Treino")
plt.plot(epochs,acuracia_teste,'-b',label="Acurácia Dados de Teste")
plt.legend()
plt.xlabel('Epochs')
plt.ylabel("Acurácia")
plt.show() 