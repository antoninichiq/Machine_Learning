from sklearn.datasets import load_iris
import pandas as pd
import tensorflow as tf
iris = load_iris()
x = pd.DataFrame(iris.data, columns=[iris.feature_names])
y = pd.Series(iris.target)

import sklearn.utils
x_shuffle, y_shuffle = sklearn.utils.shuffle(x, y)

from keras.utils import to_categorical
y_one_hot = to_categorical(y_shuffle)

from sklearn.model_selection import train_test_split

#Separando os dados entre treino e teste:
x_treino, x_teste, y_treino, y_teste = train_test_split(x_shuffle, y_one_hot, test_size =0.3, random_state=5)
#print(y_treino[:])
#print(x_treino.iloc[:])

learning_rate = 0.001
epochs = 500
batch_size = 50

n_entrada = 4
n_camada_1 = 50
n_classes = 3 # fazer o one hot encoding das 3 classes (y_treino e y_teste)

x = tf.placeholder(tf.float32,[None,n_entrada])
y = tf.placeholder(tf.float32,[None,n_classes])

w1 = tf.Variable(tf.random_normal([n_entrada,n_camada_1],stddev=(2/(n_entrada)**0.5)))
b1 = tf.Variable(tf.zeros([n_camada_1]))
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))

w_out = tf.Variable(tf.random_normal([n_camada_1,n_classes],stddev=(2/(n_camada_1)**0.5)))
b_out = tf.Variable(tf.zeros([n_classes]))
saida = tf.matmul(layer_1,w_out) + b_out

custo = tf.nn.softmax_cross_entropy_with_logits(logits=saida,labels=y)
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

predicoes = tf.equal(tf.argmax(saida,1),tf.argmax(y,1))
acuracia = tf.reduce_mean(tf.cast(predicoes,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    total_batchs = int(len(x_treino) / batch_size)
    for epoch in range(epochs):    
        custo_medio = 0.0    
        soma = 0
        for i in range(total_batchs):
            batch_x = x_treino.iloc[soma:batch_size*(i+1)]
            batch_y = y_treino[soma:batch_size*(i+1)]
            soma+=batch_size
            sess.run(otimizador,feed_dict={x:batch_x,y:batch_y})
            custo_medio += sess.run(custo,feed_dict={x:batch_x,y:batch_y})/total_batchs
            
        acuracia_teste = sess.run(acuracia,feed_dict={x:x_teste,y:y_teste})
        print("Epoch: ",epoch+1)
        print("Acucuracia teste: ","{:.3f}\n".format(acuracia_teste))
    print("Treinamento concluido!")
    print("Acuracia do Modelo: ",acuracia.eval({x:x_teste,y:y_teste}))


