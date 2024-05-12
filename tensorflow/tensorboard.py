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

#Criando um sumário para as imagens de entrada
redimensionada = tf.reshape(x,[-1,28,28,1]) #o sumary.image requer dados no formato [n_imagens,altura,base,rgb] -1 pega todas as 55mil imagens. Como vamos mexer com imagens, precisamos voltar pro formato 28x28. rgb 1 é a cor cinza
tf.summary.image('img_entrada',redimensionada,1000) #[nome,data,steps] step visualiza a primeira amostra, dps a amostra mil, dps a 2mil

def sumario_informacoes(matriz): #pode ser o peso w1 etc
    with tf.name_scope('sumario_informacoes'):
        tf.summary.scalar('media',tf.reduce_mean(matriz)) #faz a media da matriz e calcula um valor só, escalar
        tf.summary.scalar('maximo',tf.reduce_max(matriz))
        tf.summary.scalar('minimo',tf.reduce_min(matriz))
        tf.summary.histogram('histograma',matriz)
        

w1 = tf.Variable(tf.random_normal([n_entrada,n_camada_1],stddev=0.05))
sumario_informacoes(w1)
b1 = tf.Variable(tf.zeros([n_camada_1]))
sumario_informacoes(b1)
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))

w2 = tf.Variable(tf.random_normal([n_camada_1,n_camada_2],stddev=0.05))
sumario_informacoes(w2)
b2 = tf.Variable(tf.zeros([n_camada_2]))
sumario_informacoes(b2)
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,w2),b2))

w_out = tf.Variable(tf.random_normal([n_camada_2,n_classes]))
sumario_informacoes(w_out)
b_out = tf.Variable(tf.zeros([n_classes]))
sumario_informacoes(b_out)
saida = tf.matmul(layer_2,w_out) + b_out

custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=saida,labels=y))
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

predicoes = tf.equal(tf.argmax(saida,1),tf.argmax(y,1))
acuracia = tf.reduce_mean(tf.cast(predicoes,tf.float32))
tf.summary.scalar('acuracia',acuracia) #Criando um sumário escalar para a acurácia

merged = tf.summary.merge_all() #Unindo todos os sumários para rodas tudo de um vez 

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter("C:/Users/anton/OneDrive/Documentos/rede1",sess.graph)
    
    for epoch in range(epochs):
        total_batchs = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batchs):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            sess.run(otimizador,feed_dict={x:batch_x,y:batch_y})
        
        acuracia_teste = sess.run(acuracia,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        sumarios = sess.run(merged,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        writer.add_summary(sumarios,epoch) #queremos adicionar um novo sumário, não sobescrever
        
        print("Epoch: ",(epoch+1))
        print("Acuracia teste: ","{:.3f}\n".format(acuracia_teste))
    print("Treinamento concluido!")
    print("Acuracia do Modelo: ",acuracia.eval({x:mnist.test.images,y:mnist.test.labels}))
