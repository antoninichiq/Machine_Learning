import os

nomes_amostras = os.listdir("C:/Users/anton/OneDrive/Documentos/archive(12)/samples")
nomes_amostras
len(nomes_amostras)

from PIL import Image
imagens = []
for nome in nomes_amostras: #para cada nome de imagem
    endereco = "C:/Users/anton/OneDrive/Documentos/archive(12)/samples/"+ nome
    img = Image.open(endereco)
    imagens.append(img)

imagens[0]

import numpy as np

# Assuming 'imagens[0]' is your PngImageFile object
image_np = np.array(imagens[0])
print(image_np.shape) #está numa camada acima da rgb

largura,altura = imagens[0].size
print("Largura: ", largura)
print("Altura: ", altura)

#Convertendo para cinza
imagens_cinzas = []
for i in imagens:
    img_cinza = i.convert('L')
    imagens_cinzas.append(img_cinza)
    
len(imagens_cinzas)
imagens_cinzas[0]

image_np = np.array(imagens_cinzas[0])
print(image_np.shape) #converteu para (50,200)

# Vamos recortar essas imagens, cada caracter separadamente. Assim, treinamos o modelo
#a encontrar cada caracter separadamente
# Cortando uma imagem
posicoes = (30, 12, 50, 50) # Definindo o retângulo: (esquerda, cima, direita, baixo) - posições limites do retângulo
imagem_cortada = imagens_cinzas[0].crop(posicoes)
Image.open(imagem_cortada)

#Estamos com um problema multiclasses, mas cada entrada na realidade representa 5 entrada. Precisamos
#fazer com que uma entrada seja quebrada em partes, cada uma tendo uma previsao separada multiclasse
#(36 classes nesse caso). Iremos quebrar cada imagem em 5 partes, transformando cada caractere em uma 
#imagem distinta de entrada, ou seja, a dimensão (1070,50,200) vai ficar (5350,38,20)
image_np = np.array(imagem_cortada)
print(image_np.shape) #converteu para (50,200)
len(imagens_cinzas)

#Separando cada caractere em uma imagem distinta
x_novo = np.zeros((len(nomes_amostras)*5,38,20)) #(n_amostras,(dimensão))
for i in range(len(imagens_cinzas)):
    esquerda,cima,direita,baixo = 30,12,50,50
    
    for j in range(5): #para cada caractere da imagem
        posicao = (esquerda,cima,direita,baixo)
        x_novo[(i*5)+j] = imagens_cinzas[i].crop(posicao) #salva em x_novo na posicao 0 a 4, dps 5 a 9
        esquerda += 20
        direita += 20
        
imagens_cinzas[0]

# Convert the NumPy array to a PIL Image
img = Image.fromarray(x_novo[0].astype('uint8')) 
# Display the image
img.show()

x_novo.shape

x = np.zeros((x_novo.shape[0],x_novo.shape[1],x_novo.shape[2],1)) #(n_amostras, (dimensao), gray,scale)

for i in range(x_novo.shape[0]):
    norm = x_novo[i] / 250 #normalizar
    img = np.reshape(x_novo[i], (x_novo.shape[1],x_novo.shape[2],1)) #cria uma dimensao extra para indicar gray_scale
    x[i] = img #adiciona cada uma das imagens no array x
    
#Criando a varivável target
y_atual = nomes_amostras

y_atual[0][0:5]
len(y_atual)

#Precisamos separar cada um dos valores de y_atual, para ficar com uma única lista de 5350 valores
y = [None] * x.shape[0]
for i in range(len(y_atual)):
    for j in range(5):
        y[(i*5)+j] = y_atual[i][j]
#Agora temos em y todos os caracteres recortados, equivalentes às imagens que criamos em x

len(y)

#Precisamos fazer one hot enconding na variável y. Para tanto, primeiro vamos definir quais são as possíveis classes.
#Agrupando todos os símbolos em uma única string
simbolos = 'abcdefghijklmnopqrstuvwxyz' + '0123456789'

simbolos

import string
simbolos.find('f') #posição em que a letra f está

y_final = np.zeros((len(y),36)) # define a dimensao final (5350 amostras, 36 classes)
for i in range(len(y)):
    caractere = y[i]
    loc_caractere = simbolos.find(caractere)
    y_final[i,loc_caractere] = 1
    
y_final[0].dtype
y_final.shape

# Obter o caminho da pasta Documentos
documentos_path = os.path.expanduser("C:/Users/anton/OneDrive/Documentos")
# Salvar x_treino na pasta Documentos
np.save(os.path.join(documentos_path, 'x.npy'), x)
# Salvar y_treino na pasta Documentos
np.save(os.path.join(documentos_path, 'y_final.npy'), y_final)
