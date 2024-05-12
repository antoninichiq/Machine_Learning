import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True) #55mil figuras de amostra

#Parâmetros gerais
learning_rate = 0.001
epochs = 10
batch_size = 200

#Parâmetros da rede neural
n_entrada = 784 #dados de entrada MNIST (imagens de dimensão 28x28)
n_camada_1 = 30 # neurônios da primeira camada oculta
n_camada_2 = 30 # neurônio da segunda camada oculta
n_classes = 10 # total de classes MNIST (dígitos 0-9)

#Variáveis preditoras e target (emforma de placeholders)
x = tf.placeholder(tf.float32,[None,n_entrada]) #None pq depende do número de batchs
y = tf.placeholder(tf.float32,[None,n_classes])

#Pesos da camada 1
w1 = tf.Variable(tf.random_normal([n_entrada,n_camada_1],stddev=0.05)) #desvio padrao de 0.05. É extremamente relevante. Equação sugerida do desvio padrao para a função relu = 2/(número_neuronios_camada_anterior)**0.5
#Bias da camada 1
b1 = tf.Variable(tf.zeros([n_camada_1])) #inicializamos os bias com zeros (30,)
#Camada 1
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1)) #multiplica os pesos pela entra e soma o bias, aplica o relu (200x30)

#Pesos da camada 2
w2 = tf.Variable(tf.random_normal([n_camada_1,n_camada_2],stddev=0.05))
#Bias da camada 2
b2 = tf.Variable(tf.zeros([n_camada_2]))
#Camada 2
layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1,w2),b2)) #multiplica os pesos pelo resultado da camada 1, soma bias, aplica relu

#Pesos da camada de Saída (output)
w_out = tf.Variable(tf.random_normal([n_camada_2,n_classes],stddev=0.05))
#Bias da camada de Saída (output)
bias_out = tf.Variable(tf.zeros([n_classes]))
#Camada de Saída (output)
saida = tf.matmul(layer_2,w_out) + bias_out #Não colocamos a função de ativação Softmax pq a função de custo do tensorflow já possui o calculo do softmax

#Função de custo
custo = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=saida,labels=y)) #logits é o log das probabilidades não normalizadas. Labels nesse caso é o gabarito para a função de custo
#Otimizador
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

#Testando o Modelo
predicoes = tf.equal(tf.argmax(saida,1),tf.argmax(y,1)) #equal compara os resultados passados. argmax(saida,1) pega o maximo resultado entre a saida e 1 e atribui o valor de 1 pro que estiver mais proximo de 1, comparando com o neuronio mais ativado do gabarito (argmax(y,1)) 
#predicoes acumula uma lista de 0 e 1 pra ver quantos a rede acertou

#Calculando a acuracia
acuracia = tf.reduce_mean(tf.cast(predicoes,tf.float32))

#Inicializando as variváveis
init = tf.global_variables_initializer()
#Abrindo a Sessão
with tf.Session() as sess:
    sess.run(init)
    #Ciclo de treinamento
    for epoch in range(epochs):
        custo_medio = 0.0
        total_batchs = int(mnist.train.num_examples / batch_size) #total de amostras divido pelo tamanho de cada batch
        
        #Loop para todas as iterações (batches)
        for i in range(total_batchs):
            batch_x,batch_y = mnist.train.next_batch(batch_size) #extrai de 200 em 200 amostras do conjunto de dados
            
            #Fit training usando batch data
            sess.run(otimizador,feed_dict={x:batch_x,y:batch_y})
            
            #Computando o custo (Loss) média de um epoch completo (soma todos os custos de cada batch e divide pelo total de batchs)
            custo_medio += sess.run(custo,feed_dict={x:batch_x,y:batch_y}) / total_batchs
        
        #Rodando a acurácia em cada epoch
        acuracia_teste = sess.run(acuracia,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        
        #Mostrando os resultados após cada epoch
        print("Epoch: ", "{},".format((epoch+1)),"Custo medio treino = ","{:.3f}".format(custo_medio))
        print("Acuracia teste = ", "{:.3f}".format(acuracia_teste))
    print("Treinamento concluido!")
    print("Acuracia do Modelo", acuracia.eval({x:mnist.test.images,y:mnist.test.labels}))
    
    
            