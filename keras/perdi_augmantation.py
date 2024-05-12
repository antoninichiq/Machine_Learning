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

# Realizando data augmentation no Keras:
# Essa função abaixo altera cada imagem de entrada, de acordo com os parâmetros informados, variando aleatoriamente os valores.
# Em cada epoch, o conjunto de dados de treino e sempre diferente. Se eu tinha 50 mil amostras, em cada epoch terei 50 mil, mas
# todas serdo Ligeiramente diferentes das amostras da epoch anterior. Ou seja, e como se em cada epoch eu treinasse com um
# dataset diferente. Quanto mais epochs, mais "datasets" estarei usando.

from keras.preprocessing.image import ImageDataGenerator
# Documentação data augmentation: https://keras.io/preprocessing/image/
# Configurando o gerador de dados:
aug_data = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
# Passando os dados de entrada para o gerador:
treino_aumentado = aug_data.flow(x_treino_normalizado, y_treino, batch_size=200)

# Definindo o otimizador e a funcao de custo
modelo.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Treinando o modelo:
n_passadas = int(x_treino_normalizado.shape[0] / 200)
historico = modelo.fit_generator(treino_aumentado, steps_per_epoch=n_passadas, epochs=100, validation_data=(x_teste_normalizado,y_teste))

imagem = x_teste[10]
plt.imshow(imagem)
plt.show()

import numpy as np
imagem = imagem.astype('float32')
imagem - imagem / 255.0
imagem - np.expand_dims(imagem, axis=0) # criando uma dimensao extra para informar que ha apenas uma imagem por batch_size

resultado = modelo.predict_classes(imagem)
print(resultado[0])

print(resultado)