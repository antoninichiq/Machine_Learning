import pandas as pd
pd.set_option('display.max_columns',23)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (6)/recipeData.csv", encoding='latin-1')

print(arquivo.head())
print(arquivo.shape)
print(arquivo.dtypes)
faltantes = arquivo.isnull().sum()
faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['StyleID'])) * 100
print(faltantes_percentual)

pd.set_option('display.max_rows', None)
print(arquivo['StyleID'].value_counts())

arquivo = arquivo.loc[arquivo['StyleID'].isin([7,10,154,9,4,30,86,12,92,6,175,39])]

arquivo.drop(['BeerID', 'Name','URL', 'Style', 'UserId', 'PrimingMethod', 'PrimingAmount'], axis = 1, inplace = True)


print(arquivo.head())
print(arquivo.dtypes)
print("\n")
print(arquivo['SugarScale'].value_counts())
print(arquivo['BrewMethod'].value_counts())
# ----------------------------
#One hot Enconding
arquivo['SugarScale'] = arquivo['SugarScale'].replace("Specific Gravity",0)
arquivo['SugarScale'] = arquivo['SugarScale'].replace("Plato",1)

brewmethod_encode = pd.get_dummies(arquivo['BrewMethod'])
arquivo.drop('BrewMethod', axis = 1, inplace=True)
concatenando = pd.concat([arquivo, brewmethod_encode],axis=1)
# -----------------------------

print(concatenando.head())

faltantes = concatenando.isnull().sum()
faltantes_percentual = (concatenando.isnull().sum() / len(concatenando['StyleID'])) * 100
print(faltantes_percentual)


import matplotlib.pyplot as plt
# concatenando.boxplot(column = ["BoilGravity","MashThickness","PitchRate","PrimaryTemp"])
# plt.show() #outliers

concatenando.hist(column=["BoilGravity","MashThickness","PitchRate","PrimaryTemp"], bins=20)
plt.show()

concatenando["PitchRate"].fillna(concatenando["PitchRate"].mean(),inplace=True)
concatenando.fillna(concatenando.median(),inplace=True)

faltantes = concatenando.isnull().sum()
faltantes_percentual = (concatenando.isnull().sum() / len(concatenando['StyleID'])) * 100
print(faltantes_percentual)

y = concatenando['StyleID']
x = concatenando.drop('StyleID', axis = 1)

def melhor_classificacao():
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.tree import DecisionTreeClassifier #O dataset é um problema de classificação
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler # função de normalização. Cada coluna tem uma característica diferente. Uma tem ordem de grandeza na ordem de 0.02 e outra na ordem das dezenas. 
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    skfold = StratifiedKFold(n_splits=3)
    
    #Normalizando as variáveis preditoras para o KNN
    normalizador = MinMaxScaler(feature_range= (0, 1))
    X_norm = normalizador.fit_transform(x)
    
    logist= LogisticRegression()
    naive = GaussianNB()
    decision_tree = DecisionTreeClassifier()
    knn = KNeighborsClassifier()
    
    resul_logist= cross_val_score(logist,x,y,cv=skfold)
    resul_naive= cross_val_score(naive,x,y,cv=skfold)
    resul_decision_tree= cross_val_score(decision_tree,x,y,cv=skfold)
    resul_knn= cross_val_score(knn,X_norm,y,cv=skfold)
    
    dic_classmodels = {'Logistica':resul_logist.mean(), 'Naive':resul_naive.mean(), 'Decision Tree':resul_decision_tree.mean(), 'KNN':resul_knn.mean()}
    melhor_modelo = max(dic_classmodels, key=dic_classmodels.get)
    print('Regressao Logistica:', resul_logist.mean(), 'Naive Bayes:', resul_naive.mean(), 'Decision Tree:', resul_decision_tree.mean(),'KNN:',resul_knn.mean())
    print('Melhor modelo:', melhor_modelo, 'com o valor:', dic_classmodels[melhor_modelo])
        
melhor_classificacao()
    
    
def performance_melhor_classificado():
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.model_selection import GridSearchCV
    
    normalizador = MinMaxScaler(feature_range= (0, 1))
    X_norm = normalizador.fit_transform(x)
    
    valores_K = np.array([3,5,7,9])
    calculo_distancia = ['minkowski','chebyshev']
    valores_p = np.array([1,2,3])
    valores_grid = {'n_neighbors':valores_K,'metric':calculo_distancia,'p':valores_p}
    
    modelo = KNeighborsClassifier()
    
    gridKNN = GridSearchCV(estimator= modelo, param_grid=valores_grid,cv=3,n_jobs=-1)
    gridKNN.fit(X_norm,y)
    
    print("Melhor acurácia: ", gridKNN.best_score_)
    print("Melhor: ", gridKNN.best_estimator_.n_neighbors)
    print("Método distância: ", gridKNN.best_estimator_.metric)
    print("Melhor valor p: ", gridKNN.best_estimator_.p)
    
performance_melhor_classificado()

def aprenderemos_GradienteBoosting_modulo2():
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import cross_val_score
    modelo = GradientBoostingClassifier(n_estimators=300)
    skfold = StratifiedKFold(n_splits=3)
    resultado = cross_val_score(modelo, x,y,cv=skfold,n_jobs=-1)
    print(resultado.mean())

# aprenderemos_GradienteBoosting_modulo2()
