import surprise #bilioteca especializada para trabalhar com sistema de recomendação
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (13)/ratings.txt',sep=" ",names = ['id_usuario','id_filme','rating'])

print(dataset.head(20))

filmes = len(dataset['id_filme'].unique()) # pega só os valores únicos e guarda lista, usei isso pq pode repetir o mesmo filme mais de uma vez pra mais de um usuario
usuarios = len(dataset['id_usuario'].unique())
rating = dataset.shape[0]
print('Total de filmes: ',filmes)
print("Total de usuários: ",usuarios)
print("Total de ratings: ",rating)

dataset['rating'].value_counts().plot(kind='bar')
plt.show()

menor_rating = dataset['rating'].min()
maior_rating = dataset['rating'].max()

print(f'Variação de rating: {menor_rating} a {maior_rating}')

# Definindo o range de ratings
reader = surprise.Reader(rating_scale=(0.5,4.0))
dataset_surprise = surprise.Dataset.load_from_df(dataset,reader) # funçao necessária para passar o dataset para o surprise

# Escolhendo o algoritmo e trainando o modelo
dataset_preenchido = dataset_surprise.build_full_trainset()
algoritmo = surprise.SVDpp(n_factors = 20)
algoritmo.fit(dataset_preenchido)

dataset_missing = dataset_preenchido.build_anti_testset() # previsões para os dados que não estão preenchidos
previsoes = algoritmo.test(dataset_missing) # o usuario x avaliou 5 filmes, mas tem 2000 e tantos

print(previsoes)
len(previsoes)

# Agora, vamos verificar as top recomendações pra cada usuario
from collections import defaultdict # cria um dicionario vazio

def obtem_top_n(previsoes,n=5): # 5 melhores recomendações
    top_n = defaultdict(list) # cria um dicionario onde os valores são listas vazias
    for usuario, filme,_,previsao,_ in previsoes:
        top_n[usuario].append((filme,previsao))
    for usuario, previsoes_usuario in top_n.items():
        previsoes_usuario.sort(key=lambda x: x[1],reverse=True) # ordena as previsoes de rating do maior para o menor
        top_n[usuario] = previsoes_usuario[:n]
    return top_n

top_5 = obtem_top_n(previsoes)
print(top_5)

for usuario, previsoes_usuario in top_5.items():
    print(usuario, [filme for (filme,_) in previsoes_usuario])
    
    
# Fazendo uma previsão somente para um usuário e filme específicos:
previsao_usuario = algoritmo.predict(uid='30', iid='87')
rating = previsao_usuario.est
print(rating)


# Validando o modelo
from surprise.model_selection import train_test_split
dataset_treino, dataset_teste = train_test_split(dataset_surprise,test_size=0.3)

algoritmo = surprise.SVDpp(n_factors = 20)
algoritmo.fit(dataset_treino)
previsoes_gerais = algoritmo.test(dataset_teste)

print(previsoes_gerais)

from surprise import accuracy
print(accuracy.rmse(previsoes_gerais))


# Ajustando os parametros
param_grid = {'lr_all': [.007, .01, 0.05, 0.001], 'reg_all': [0.02, 0.1, 1.0, 0.005]}
surprise_grid = surprise.model_selection.GridSearchCV(surprise.SVDpp, param_grid, measures=['rmse'], cv=3, n_jobs =- 1)
surprise_grid.fit(dataset_surprise)
print(surprise_grid.best_params['rmse'])


#Vamos ver agr os filmes mais semelhantes entre si... quais sao os usuario vizinhos entre si?
#Vamos usar o conceito de cosine distance
# Mostrando os dados vizinhos
from surprise import KNNBasic

dataset_preenchido = dataset_surprise.build_full_trainset()
algoritmo - KNNBasic(sim_options={'name': 'cosine', 'user_based': False})
# 'name' é o algoritmo de similaridade, 'user_based' == True é para calcular a similaridade entre usuários
algoritmo.fit(dataset_preenchido)

# Mostrando os k vizinhos mais próximos:
vizinhos = algoritmo.get_neighbors(343, k=10)

print('Os 10 filmes vizinhos para o id escolhido são:')
for filme in vizinhos:
    print(filme)