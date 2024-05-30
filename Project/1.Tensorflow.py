import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) #55mil figuras de amostra

learning_rate = 0.001
epochs = 10
batch_size = 200

n_entrada = 784
n_classes = 10

x = tf.placeholder(tf.float32,[None,n_entrada])
y = tf.placeholder(tf.float32,[None,n_classes])

def conv2d(entrada,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(entrada,w,strides=[1,1,1,1],padding='VALID'),b))
#Nessa linha de código, primeiro adicionamos um feature map multiplicando a entra pelos pesos w, depois somamos bias e
#aplicamos relu em cima disso.
#strides = [',deslocamento,_horizontal,deslocamento_vertical,1] #doumentacao pede para cravar em 1 os extremos

def max_pool(entrada,k): # Aentrada é o feature map criado acima 
    return tf.nn.max_pool(entrada,ksize=[1,k,k,1],strides=[1,k,k,1],padding='VALID')

x_redimensionado = tf.reshape(x,shape=[-1,28,28,1])
#O modelo Conv2d espera que na entrada eu passe 4 dimensões: amostras, altura, largura e padrao de cores

w1 = tf.Variable(tf.random_normal([5,5,1,32],stddev=0.05)) #kernel 5x5 (filtro), 1 imagem para varrer (entrada), 32 feature maps(saída)
b1 = tf.Variable(tf.zeros([32])) #1 bias para cada feature map da camada 1
                 
w2 = tf.Variable(tf.random_normal([5,5,32,64],stddev=0.05)) #5x5, 32 imagens para varrer (pooling layers do layer 1), 64 features (saída) 
b2 = tf.Variable(tf.zeros([64])) #1 bias para cada feature m ap da camada 2

w_denso_1 = tf.Variable(tf.random_normal([4*4*64,80],stddev=0.0063)) #80 neurônios se conectam a 1024 neurônios (saída anterior)
b_denso_1 = tf.Variable(tf.zeros([80])) #1 bias para cada neurônio da camada densa

w_out = tf.Variable(tf.random_normal([80,n_classes],stddev=0.04)) #10 neurônios (saída) se conectam a 80 neurônios (entrada)
b_out = tf.Variable(tf.zeros([n_classes])) #1 bias para cada neurônio correspondente a uma classe3

#Camada CNN 1
conv1 = conv2d(x_redimensionado,w1,b1)
pool1 = max_pool(conv1,k=2)
#drop1 = tf.nn.dropout(pool1,rate=0.2)

#Camada CNN 2
conv2 = conv2d(pool1,w2,b2)
pool2 = max_pool(conv2,k=2)
#drop2 = tf.nn.dropout(pool2,rate=0.2)

#Camada densa oculta
pool2_redimensionada = tf.reshape(pool2,shape=[-1,w_denso_1.get_shape().as_list()[0]]) #Estamos pegando o valor 1024 do w_denso_1. Estamos aplicando o Flatten (transformar em vetor) antes de ligar na camada dense
densa = tf.nn.relu(tf.add(tf.matmul(pool2_redimensionada,w_denso_1),b_denso_1))
pool_densa = tf.nn.dropout(densa,keep_prob=0.2)

#Camada de saída
out = tf.add(tf.matmul(pool_densa,w_out),b_out)

#Custo e Otimizador
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y))
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

acertos = tf.equal(tf.argmax(out,1),tf.argmax(y,1))
acuracia = tf.reduce_mean(tf.cast(acertos,tf.float32))

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
    print("Treinamento concluido!")
    print("Acuracia do Modelo: ",acuracia.eval({x:mnist.test.images,y:mnist.test.labels}))
