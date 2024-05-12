import numpy as np
import tensorflow as tf
import time

#Criando as matrizes
matriz_a = np.random.rand(10000,10000).astype('float32')
matriz_b = np.random.rand(10000,10000).astype('float32')

#Armazenando os resultados
resultados = []

#Criando uma função que multiplica uma matriz por ela mesma
def mul_matrizes(matriz):
    return tf.matmul(matriz,matriz)

#Definindo o dispositvo que fará as multiplicações
with tf.device('/cpu:0'): #poderiamos usar a cpu
    a = tf.placeholder(tf.float32,[10000,10000])
    b = tf.placeholder(tf.float32,[10000,10000])
    resultados.append(mul_matrizes(a))
    resultados.append(mul_matrizes(b))
    
#Somando os resultados (usando CPU)
with tf.device('/cpu:0'):
    soma = tf.add_n(resultados)

#Comecando a contar o tempo
inicio = time.time()

#Sessão
with tf.Session() as sess:
    sess.run(soma,{a:matriz_a,b:matriz_b})

#Terminando a contagem do tempo
fim = time.time()

#Imprimendo o tempo de execucao
print("Tempo em segundos: ", fim-inicio)
