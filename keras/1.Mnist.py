from keras.datasets import mnist
(x_treino,y_treino),(x_teste,y_teste) = mnist.load_data()

print(x_treino.shape)

from keras.utils import to_categorical
y_treino = to_categorical(y_treino)
y_teste = to_categorical(y_teste)

#O modelo Conv2D espera que na entrada eu passe 3 dimenões: altura, largura e padrao de cores
x_treino = x_treino.reshape(60000,28,28,1) #1 é a escala de cinza, poderia ser rgb
x_teste = x_teste.reshape(10000,28,28,1)

from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, Flatten, MaxPooling2D

#Criando a rede CNN
modelo = Sequential()
#criando a primeira camada de convolucao
modelo.add(Conv2D(filters=32,kernel_size=5,activation='relu',input_shape=(28,28,1))) #filters = feature_maps, kernel_size é o tamanho do filtro
modelo.add(MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='valid')) #pool_size = (linha,coluna). stride=None, preeche automaticamente pra (2,2). padding='valid' significa que nao tem padding, pra ativar deveria ser 'same'
modelo.add(Conv2D(filters=64,kernel_size=5,activation='relu'))
modelo.add(MaxPooling2D(pool_size=(2,2),strides=None,padding='valid'))
modelo.add(Flatten()) #essa função transforma um espaço multidimension em uma única dimensão. Por exemplo, uma matriz de duas
#dimensoes 28x28 seria transformada em uma única dimensão de tamanho 784. É como se cada linha da matriz fosse colocada uma do lado
#da outra, ficando um array longo de uma única dimensão.
modelo.add(Dense(80,kernel_initializer='normal',activation='relu'))
modelo.add(Dropout(0.2))
modelo.add(Dense(10,kernel_initializer='normal',activation='softmax'))

modelo.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

modelo.fit(x_treino,y_treino,batch_size=200,epochs=10,validation_data=(x_teste,y_teste),verbose=1)







