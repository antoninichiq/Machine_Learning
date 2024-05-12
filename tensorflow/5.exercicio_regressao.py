import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import pandas as pd
arquivo = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (3)/Admission_Predict.csv')
arquivo.drop('Serial No.', axis=1, inplace=True)
#Separando as variáveis entre preditoras e variável target
y = arquivo['Chance of Admit ']
x = arquivo.drop('Chance of Admit ', axis = 1)

print(y.shape)

y_remodelado = y.values.reshape(400,1)

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y_remodelado, test_size = 0.3, shuffle= True)

print(x_treino.shape) #280 amostras #7colunas
print(y_treino.shape) #280 amostras #1 coluna

learning_rate = 0.001
epochs = 500
batch_size = 50

n_entrada = 7
n_camada_1 = 10
n_classes = 1

x = tf.placeholder(tf.float32,[None,n_entrada])
y = tf.placeholder(tf.float32,[None,n_classes])

w1 = tf.Variable(tf.random_normal([n_entrada,n_camada_1],stddev=( 2 / ( n_entrada ) ** 0.5)))
b1 = tf.Variable(tf.zeros([n_camada_1]))
layer_1 = tf.nn.relu(tf.add(tf.matmul(x,w1),b1))

w_out = tf.Variable(tf.random_normal([n_camada_1,n_classes],stddev=( 2 / ( n_camada_1 ) ** 0.5)))
b_out = tf.Variable(tf.zeros([n_classes]))
saida = tf.matmul(layer_1,w_out) + b_out

custo = tf.reduce_mean(tf.losses.mean_squared_error(predictions=saida,labels=y)) 
otimizador = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(custo)

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
            
        mse = sess.run(custo,feed_dict={x:x_teste,y:y_teste})
        print("Epoch: ",epoch+1)
        print("MSE teste: ","{:.3f}\n".format(mse))
    print("Treinamento concluido!")
    print("MSE final: ",custo.eval({x:x_teste,y:y_teste}))