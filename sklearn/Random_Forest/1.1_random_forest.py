def processamento_dados():
    import pandas as pd
    pd.set_option('display.max_columns',81)
    arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (7)/BRAZIL_CITIES.csv",sep=';',decimal=',')
    print(arquivo.head())

    arquivo.drop(['CITY', 'IDHM Ranking 2010', 'IDHM_Renda', 'IDHM_Longevidade', 'IDHM_Educacao', 'LONG',
    'LAT', 'GVA_MAIN', 'REGIAO_TUR', 'MUN_EXPENDIT', 'HOTELS', 'BEDS', 'Pr_Agencies', 'Pu_Agencies', 'Pr_Bank',
    'Pu_Bank', 'Pr_Assets', 'Pu_Assets', 'UBER', 'MAC', 'WAL-MART','AREA'], axis=1, inplace=True)

    print(arquivo.head())
    print(arquivo.dtypes)

    #Transformando variáveis categóricas em números (one hot encode)
    estado_encode = pd.get_dummies(arquivo['STATE'])
    rural_encode = pd.get_dummies(arquivo['RURAL_URBAN'])
    categoria_encode = pd.get_dummies(arquivo['CATEGORIA_TUR'])

    arquivo.drop('STATE',axis=1,inplace=True)
    arquivo.drop('RURAL_URBAN',axis=1,inplace=True)
    arquivo.drop('CATEGORIA_TUR',axis=1,inplace=True)

    #Inserindo as variáveis one hot encode no dataset
    concatenar = pd.concat([arquivo,estado_encode,rural_encode,categoria_encode],axis=1)

    concatenar['IDHM'] = pd.to_numeric(concatenar['IDHM'])
    concatenar['ALT'] = pd.to_numeric(concatenar['ALT'])

    concatenar['GVA_AGROPEC'] = pd.to_numeric(concatenar['GVA_AGROPEC'])
    concatenar['GVA_INDUSTRY'] = pd.to_numeric(concatenar['GVA_INDUSTRY'])
    concatenar['GVA_SERVICES'] = pd.to_numeric(concatenar['GVA_SERVICES'])
    concatenar['GVA_PUBLIC'] = pd.to_numeric(concatenar['GVA_PUBLIC'])

    concatenar['TAXES'] = pd.to_numeric(concatenar['TAXES'])
    concatenar['GDP'] = pd.to_numeric(concatenar['GDP'])
    concatenar['GDP_CAPITA'] = pd.to_numeric(concatenar['GDP_CAPITA'])
    # remover coluna errada
    for k,v in concatenar.items():
        a = k.strip()
        if a == "GVA_TOTAL":
            concatenar.drop(k,axis=1,inplace=True)
            
    print(concatenar.head())
    pd.set_option('display.max_rows',None)
    print(concatenar.dtypes)

    concatenar = concatenar.dropna()

    faltantes = concatenar.isnull().sum()
    faltantes_percentual = (concatenar.isnull().sum() / len(concatenar['IDHM'])) * 100
    print(faltantes_percentual)
    return concatenar
concatenar = processamento_dados()
y = concatenar['IDHM']
x = concatenar.drop('IDHM', axis = 1)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

normalizador = MinMaxScaler(feature_range=(0,1))
x_norma = normalizador.fit_transform(x)

# reduzindo 81 dimensões (colunas) para 15 dimensões
pca = PCA(n_components=15)
x_pca = pca.fit_transform(x_norma)
print("Variância explicada dos componentes:", pca.explained_variance_ratio_)
print(f"Essas componentes explicam {(sum(pca.explained_variance_ratio_)):.2f}% dos dados") 

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
modelo = RandomForestRegressor(n_estimators=50,n_jobs=-1) # quantas árvores de decisão serão construídas
kfold = KFold(n_splits=5)
resultado = cross_val_score(modelo,x_pca,y,cv=kfold,n_jobs=-1)
print(resultado.mean())

import numpy as np
from sklearn.model_selection import RandomizedSearchCV
#Definindo os valores que serão testados em RandomForest
minimos_split = np.array([2,3,4])
maximo_nivel = np.array([3,5,7,9,11,14])
minimo_leaf = np.array([3,4,5,6])
valores_grid = {'min_samples_split':minimos_split,'max_depth':maximo_nivel,'min_samples_leaf':minimo_leaf}

gridRandomForest = RandomizedSearchCV(estimator= modelo, param_distributions=valores_grid,cv=3,n_iter=50,n_jobs=-1)
gridRandomForest.fit(x_pca,y)

print("Mínimo split: ", gridRandomForest.best_estimator_.min_samples_split)
print("Máxima profundidade: ", gridRandomForest.best_estimator_.max_depth)
print("Mínimo leaf: ", gridRandomForest.best_estimator_.min_samples_leaf)
print("R2: ", gridRandomForest.best_score_)
