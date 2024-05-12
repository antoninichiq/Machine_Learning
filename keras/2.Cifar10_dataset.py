from keras.datasets import cifar10
(x_treino,y_treino),(x_teste,y_teste) = cifar10.load_data()

import matplotlib.pyplot as plt
def mostrar():
    for i in range(9):
        #mostrará 9 imagens na mesma figura, dividias em 3 linhas e 3 colunas
        plt.subplot(3,3,i+1)#(linhas,colunas,indice). Esse índice é gerado automaticamentw a partir das imagens que forncemos para alocá-las na tela
        plt.imshow(x_treino[i])#informa qual imagem deve mostrar considerando o índice no dataset
    plt.show()

print(x_treino.shape) #(50000,32,32,3) dimensao 32pixels x 32 pixels. 3 canais de cores RGB, que variam de 0 a 255. Teremos que normalizar
print(y_treino.shape)
print(y_treino[:])

from keras.utils import to_categorical
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)
print(y_treino.shape)

x_treino_float = x_treino.astype('float32')
x_teste_float = x_teste.astype('float32')

x_treino_normalizado = x_treino_float / 255.0
x_teste_normalizado = x_teste_float / 255.0

from keras.models import Sequential
from keras.layers import Dense, Dropout,Conv2D,Flatten,MaxPooling2D

#Podemos forçar o Keras a usar a inicialização Xavier, basta informar em kernel_initializar='glorot_uniform"
modelo = Sequential()
modelo.add(Conv2D(filters=32,kernel_size=5,activation='relu',input_shape=(32,32,3)))
modelo.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='same'))
modelo.add(Conv2D(filters=64, kernel_size=5, activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='same'))
modelo.add(Flatten()) # Transformando um espaco multidimensional em uma única dimensão
modelo.add(Dense(80, kernel_initializer='glorot_uniform', activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(10, kernel_initializer='glorot_uniform', activation="softmax"))

modelo.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
historico = modelo.fit(x_treino_normalizado,y_treino,batch_size=200,epochs=30,validation_data=(x_teste_normalizado,y_teste))

acuracia_treino = historico.history['accuracy']
acuracia_teste = historico.history['val_accuracy']

epochs = range(1, len(acuracia_treino)+1)

plt.plot(epochs, acuracia_treino, '-g', label='Acuracia Dados de Treino')
plt.plot(epochs, acuracia_teste, '-b', label='Acuracia Dados de Teste')
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Acurácia")
plt.show()



