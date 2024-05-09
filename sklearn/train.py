import pandas as pd 
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
import numpy as np
def processar_dataset():
    pd.set_option('display.max_columns',None)
    arquivo1 = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (14)/0.csv',header=None)
    arquivo2 = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (14)/1.csv',header=None)
    arquivo3 = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (14)/2.csv',header=None)
    arquivo4 = pd.read_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/archive (14)/3.csv',header=None)
    pd.set_option('display.max_rows',None)
    arquivo = pd.concat([arquivo1,arquivo2,arquivo3,arquivo4],axis=0)
    print(arquivo.head())
    print(arquivo.dtypes)
    faltantes_percentual = (arquivo.isnull().sum() / len(arquivo[64])) * 100
    print(faltantes_percentual)
    arquivo.to_csv('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/arquivo_concatenado.csv', index=False)
    return arquivo

arquivo = pd.read_excel('C:/Users/anton/OneDrive/Área de Trabalho/Curso - ML/Datasets/arquivo_concatenado.xlsx')
y = arquivo[64]
x = arquivo.drop(64,axis=1)

def stratifiedkfold(modelo,x,y):
    from sklearn.model_selection import StratifiedKFold,cross_val_score
    skfold = StratifiedKFold(n_splits=3,shuffle=True)
    resultado = cross_val_score(modelo,x,y,cv=skfold,n_jobs=-1)
    print(resultado.mean())

def logisitc_regression(stratfied=False,random=False,bagging=False):
    from sklearn.linear_model import LogisticRegression
    
    modelo = LogisticRegression(max_iter=1000)
    
    if stratfied:
        stratifiedkfold(modelo,x,y)
        
    if bagging:
        from sklearn.ensemble import BaggingClassifier
        modelo = BaggingClassifier(estimator=LogisticRegression(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
        stratifiedkfold(modelo,x,y)
    if random:
        valores_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 5, 10, 20, 50, 100])
        regularizacao = ['l1','l2']
        valores_grid = {'C':valores_C, 'penalty':regularizacao}
        
        procura = RandomizedSearchCV(estimator=modelo, param_distributions=valores_grid,cv=3,random_state=15,n_jobs=-1)
        procura.fit(x,y)

        print("Melhor acurácia: ",procura.best_score_)
        print("Parametro C: ", procura.best_estimator_.C)
        print("Regularizacao: ", procura.best_estimator_.penalty)
# logisitc_regression(bagging=True) # acurácia de 0.358 com RandomizedSearchCV e Bagging

def gaussianNB():
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    #Criação do modelo:
    modelo = GaussianNB()
    kfold = KFold(n_splits=3)
    resultado = cross_val_score(modelo,x,y,cv = kfold, n_jobs =- 1)
    print(resultado.mean())
# gaussianNB() # acuracia de 0.44
 
def knn(stratified=False,random=False,grid=False,bagging=False):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    
    normalizador = MinMaxScaler(feature_range=(0,1))
    x_norm = normalizador.fit_transform(x)
    
    modelo = KNeighborsClassifier()
    
    if stratified:
        stratifiedkfold(modelo,x_norm,y)
    
    if bagging:
        from sklearn.ensemble import BaggingClassifier
        modelo = BaggingClassifier(estimator=KNeighborsClassifier(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
        stratifiedkfold(modelo,x_norm,y)

    if random:
        valores_K = np.array([3,5,7,9,11])
        calculo_distancia = ['minkowski','chebyshev']
        valores_p = np.array([1,2,3,4])
        valores_grid = {'n_neighbors':valores_K, 'metric': calculo_distancia,'p':valores_p}
        
        procura = RandomizedSearchCV(estimator=modelo,param_distributions=valores_grid,cv=3,n_jobs=-1)
        procura.fit(x_norm,y)
        
        print("Melhor acurácia: ",procura.best_score_)
        print("Melhor K: ", procura.best_estimator_.n_neighbors)
        print("Método distância: ", procura.best_estimator_.metric)
        print ("Melhor valor p: ", procura.best_estimator_.p)
        
    if grid:
        valores_K = np.array([3,5,7,9,11])
        calculo_distancia = ['minkowski','chebyshev']
        valores_p = np.array([1,2,3,4])
        valores_grid = {'n_neighbors':valores_K, 'metric': calculo_distancia,'p':valores_p}
        
        #Criando os grids:
        gridKNN = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv = 5,n_jobs=-1)
        gridKNN.fit(x_norm,y)

        #Imprimindo os melhores parâmetros:
        print ("Melhor acurácia: ", gridKNN.best_score_)
        print ("Melhor K: ", gridKNN.best_estimator_.n_neighbors)
        print ("Método distância: ", gridKNN.best_estimator_.metric)
        print ("Melhor valor p: ", gridKNN.best_estimator_.p)
# knn(bagging=True) # acurácia de 0.65 com GridSearcCV e Bagging

def decision_tree(stratified=False,random=False,grid=False,bagging=False):
    from sklearn.tree import DecisionTreeClassifier
    
    modelo = DecisionTreeClassifier()
    
    if stratified:
        stratifiedkfold(modelo,x,y)
    
    if bagging:
        from sklearn.ensemble import BaggingClassifier
        modelo = BaggingClassifier(estimator=DecisionTreeClassifier(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
        stratifiedkfold(modelo,x,y)
        
    if random:
        minimos_split = np.array([2,3,4,5,6,7,8,15,20]) #cortar as variaveis que tem varios valores, pra decidir um norte pra arvore. Ex B<= 25
        maximo_nivel = np.array([8,15,17,19,21,25]) #profundidade máxima
        algoritimo = ['gini','entropy']
        valores_grid = {'min_samples_split': minimos_split, 'max_depth':maximo_nivel,'criterion':algoritimo}

        procurar = RandomizedSearchCV(estimator=modelo,param_distributions=valores_grid,cv=3,n_jobs=-1)
        procurar.fit(x,y)

        print("Mínimo split: ", procurar.best_estimator_.min_samples_split)
        print("Máxima profundidade: ", procurar.best_estimator_.max_depth)
        print ("Algoritmo escolhido: ", procurar.best_estimator_.criterion)
        print ("Acurácia: ", procurar.best_score_)
    if grid:
        minimos_split = np.array([8,9,10]) #cortar as variaveis que tem varios valores, pra decidir um norte pra arvore. Ex B<= 25
        maximo_nivel = np.array([15,16,17]) #profundidade máxima
        algoritimo = ['gini','entropy']
        valores_grid = {'min_samples_split': minimos_split, 'max_depth':maximo_nivel,'criterion':algoritimo}
        
        #Criando os grids
        gridDecisionTree = GridSearchCV(estimator= modelo, param_grid=valores_grid,cv=3,n_jobs=-1)
        gridDecisionTree.fit(x,y)

        print("Mínimo split: ", gridDecisionTree.best_estimator_.min_samples_split)
        print("Máxima profundidade: ", gridDecisionTree.best_estimator_.max_depth)
        print ("Algoritmo escolhido: ", gridDecisionTree.best_estimator_.criterion)
        print ("Acurácia: ", gridDecisionTree.best_score_)    
# decision_tree(random=True) # acurácia de 0.777 com GridSearcCV e 0.9 com Bagging

def kmeans():
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.cluster import KMeans
    
    normalizador = MinMaxScaler(feature_range= (0,1))
    x_norm = normalizador.fit_transform(x)

    modelo = KMeans(n_clusters=4,random_state=16,n_init=10)
    modelo.fit(x_norm)
    
    from sklearn.metrics import accuracy_score
    clusters = modelo.predict(x_norm)
    print(accuracy_score(y,clusters))
# kmeans() # acurácia de 0.25

def gradient_boosting(stratified=False,random=False,grid=False,bagging=False):
    from sklearn.ensemble import GradientBoostingClassifier
    if stratified:
        modelo = GradientBoostingClassifier()
        stratifiedkfold(modelo,x,y)
    
    if bagging:
        from sklearn.ensemble import BaggingClassifier
        modelo = BaggingClassifier(estimator=GradientBoostingClassifier(),n_estimators=50,max_samples=0.5 ,bootstrap=True,n_jobs=-1) #n_estimator é a qntd de arvores de decisões, max_samples é a porcentagem do conjunto de dados, bootstrap é pq pode repetir os dados
        stratifiedkfold(modelo,x,y)
        
    if random:
        minimos_split = np.array([2,3,4,5,6])
        maximo_nivel = np.array([3,4,5,6,7])
        minimo_leaf = np.array([2,3,4,5,6])
        valores_grid = {'min_samples_split':minimos_split,'max_depth':maximo_nivel,'min_samples_leaf':minimo_leaf}

        procurar = RandomizedSearchCV(estimator=modelo,param_distributions=valores_grid,cv=3,n_jobs=-1)
        procurar.fit(x,y)

        print("Mínimo split: ", procurar.best_estimator_.min_samples_split)
        print("Máxima profundidade: ", procurar.best_estimator_.max_depth)
        print('Mínimo leaf: ', procurar.best_estimator_.min_samples_leaf)
        print("Acurácia: ", procurar.best_score_) 
    if grid:
        #Definindo os valores que serão testados em GradientBoost
        criterion = ['friedman_mse', 'mse']
        max_features = np.array([64, 6, 8, 12, 16, 32])
        valores_grid ={'max_features': max_features, 'criterion' :criterion}

        #Criação do modelo:
        modelo = GradientBoostingClassifier(n_estimators=500, learning_rate=0.085, min_samples_split=3, min_samples_leaf=4, max_depth=5)

        gridGradient = GridSearchCV(estimator = modelo, param_grid = valores_grid, cv=3, n_jobs =- 1)
        gridGradient.fit(x,y)

        #Imprimindo os melhores parâmetros:
        print ("max_features: ", gridGradient.best_estimator_.max_features)
        print ("criterion: ", gridGradient.best_estimator_.criterion)
        print ("Acurácia: ", gridGradient.best_score_)
# gradient_boosting(grid=True) # acuracia de 0.944
