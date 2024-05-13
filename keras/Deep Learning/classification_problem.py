import pandas as pd
arquivo = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (12)/balance-scale.data',
                        names=['Class', 'Left-Weight', 'Left-Distance', 'Right-Weight', 'Right-Distance'])

print(arquivo.head())

y = arquivo["Class"]
x = arquivo.drop("Class",axis=1)

y_encode = pd.get_dummies(arquivo['Class']) #Temos nomes ao inves de número nessa coluna, por isso fazemos o one hot encoding
print(y_encode.head())
arquivo = pd.concat([arquivo, y_encode], axis = 1)
arquivo.drop('Class', axis = 1, inplace = True)
print(arquivo.head())
#Poderiamos fazer o one hot encoding de outra maneira
# y.replace('L',0,inplace=True) #coluna 0
# y.replace('R',1,inplace=True)
# y.replace('B',2,inplace=True)
# from keras.utils import to_categorical
# y_encode = to_categorical(y)

from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste = train_test_split(x,y_encode,test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout
from keras.optimizers import Adam

modelo = Sequential()
modelo.add(Dense(50,input_dim=4,kernel_initializer='normal',activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(50,kernel_initializer='normal',activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(50,kernel_initializer='normal',activation='relu'))
modelo.add(Dropout(0.2)) 
modelo.add(Dense(3,kernel_initializer='normal',activation='softmax'))

otimizador = Adam(amsgrad=True)

modelo.compile(loss='categorical_crossentropy',optimizer=otimizador,metrics=['acc'])
historico = modelo.fit(x_treino,y_treino,epochs=20,batch_size=10,validation_data=(x_teste,y_teste),verbose=1)

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
