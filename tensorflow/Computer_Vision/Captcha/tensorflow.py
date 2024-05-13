import numpy as np
import os
# Caminho para a pasta Documentos
documentos_path = os.path.expanduser("C:/Users/anton/OneDrive/Documentos")
# Carregar x
x_final = np.load(os.path.join(documentos_path, 'x.npy'))
# Carregar y_treino
y_final = np.load(os.path.join(documentos_path, 'y_final.npy'))

print(x_final.dtype)
print(y_final.dtype)

from sklearn.model_selection import train_test_split
x_treino,x_teste,y_treino,y_teste = train_test_split(x_final,y_final,test_size=0.2,random_state=8)

print(x_treino.shape)
print(len(x_treino))

import tensorflow as tf 
learning_rate = 0.001
epochs = 200
batch_size = 428

x = tf.placeholder(tf.float32,[None,38,20,1])
y = tf.placeholder(tf.float32,[None,36]) #36 classes, todo alfabeto minusculo + digitos de 0 a 9

def conv2d(entrada,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entrada,w,strides=[1,1,1,1],padding='VALID'),b))

def max_pool(entrada,k):
    return tf.nn.max_pool(entrada,ksize=[1,k,k,1],strides=[1,k,k,1],padding='VALID')

w1 = tf.Variable(tf.random_normal([3,3,1,32],stddev=0.05)) #kernel 3x3, 1 imagem para varrer (entrada), 32 feature maps (saída)
b1 = tf.Variable(tf.zeros([32]))

w2 = tf.Variable(tf.random_normal([3,3,32,64]))
b2 = tf.Variable(tf.zeros([64]))

w_denso = tf.Variable(tf.random_normal([16*7*64,80],stddev=0.035))
b_denso = tf.Variable(tf.zeros([80]))

w_out = tf.Variable(tf.random_normal([80,36],stddev=0.05))
b_out = tf.Variable([36])
b_out = tf.cast(b_out, tf.float32)

conv1 = conv2d(x,w1,b1)
pool1 = max_pool(conv1,k=2)

conv2 = conv2d(pool1,w2,b2)

drop2_redimensionada = tf.reshape(conv2, shape=[-1,w_denso.get_shape().as_list()[0]]) #Flatten
densa = tf.nn.relu(tf.add(tf.matmul(drop2_redimensionada,w_denso),b_denso))

out = tf.add(tf.matmul(densa,w_out),b_out)

custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y))
otimizador = tf.train.AdadeltaOptimizer(learning_rate=learning_rate).minimize(custo)

acertos = tf.equal(tf.argmax(out,1),tf.argmax(y,1))
acuracia = tf.reduce_mean(tf.cast(acertos,tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        total_batchs = int(x_treino.shape[0] / batch_size)
        soma = 0
        
        for i in range(total_batchs):
            batch_x = x_treino[soma:batch_size*(i+1)]
            batch_y = y_treino[soma:batch_size*(i+1)]
            soma+=batch_size
            sess.run(otimizador,feed_dict={x:batch_x,y:batch_y})
        
        acuracia_teste = sess.run(acuracia,feed_dict={x:x_teste,y:y_teste})
        
        print("Epoch: ",(epoch+1))
        print("Acuracia teste: ","{:.3f}\n".format(acuracia_teste))
    print("Treinamento concluido!")
    print("Acuracia do Modelo: ",acuracia.eval({x:x_teste,y:y_teste}))


