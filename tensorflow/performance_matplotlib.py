import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)

learning_rate = 0.001
epochs = 10
batch_size = 200

n_entrada = 784
n_camada_1 = 30
n_camada_2 = 30
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_entrada])
y = tf.placeholder(tf.float32,[None,n_classes])

w1 = tf.Variable(tf.random_normal([n_entrada,n_camada_1],stddev=0.05))
b1 = tf.Variable(tf.zeros([n_camada_1]))
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))
#Dropout
drop1 = tf.nn.dropout(layer_1,keep_prob=0.2)

w2 = tf.Variable(tf.random_normal([n_camada_1,n_camada_2],stddev=0.05))
b2 = tf.Variable(tf.zeros([n_camada_2]))
layer_2 = tf.nn.relu(tf.add(tf.matmul(drop1,w2),b2))
drop2 = tf.nn.dropout(layer_2,keep_prob=0.2)

w_out = tf.Variable(tf.random_normal([n_camada_2,n_classes]))
b_out = tf.Variable(tf.zeros([n_classes]))
saida = tf.matmul(drop2,w_out) + b_out

custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=saida,labels=y))
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

predicoes = tf.equal(tf.argmax(saida,1),tf.argmax(y,1))
acuracia = tf.reduce_mean(tf.cast(predicoes,tf.float32))

# Listas que armazenam os valores ao longo do treinamento para plotar no gráfico
historico_acc = []
historico_epochs = []

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        total_batchs = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batchs):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(otimizador,feed_dict={x:batch_x,y:batch_y})
        
        acuracia_teste = sess.run(acuracia,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("Epoch: ",(epoch+1))
        print("Acuracia teste: ","{:.3f}\n".format(acuracia_teste))
        historico_acc.append(acuracia_teste)
        historico_epochs.append(epoch+1)
    print("Treinamento concluido!")
    print("Acuracia do Modelo: ",acuracia.eval({x:mnist.test.images,y:mnist.test.labels}))
    
import matplotlib.pyplot as plt
plt.plot(historico_epochs,historico_acc,'-',label='Acurácia da Rede Neural')
plt.ylabel("Acurácia")
plt.xlabel("Epoch")
plt.legend()
plt.show()
