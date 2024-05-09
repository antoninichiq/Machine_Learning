def processamento_dados():
    import pandas as pd
    pd.set_option('display.max_columns',23)
    arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (6)/recipeData.csv",encoding='ISO-8859-1')
    print(arquivo.shape)

    selecao = arquivo.loc[arquivo['StyleID'].isin([7,10,134,9,4,30,86,12,92,6,175,39])]
    selecao.drop(['BeerID', 'PrimingAmount' ,'PrimingMethod', 'UserId', 'Style', 'URL', 'Name'],axis=1,inplace=True)
    print(selecao.head())

    selecao['SugarScale'] = selecao['SugarScale'].replace('Specific Gravity', 0)
    selecao['SugarScale'] = selecao['SugarScale'].replace('Plato', 1)

    #Transformando variáveis texto na coluna 'BrewMethod' em categorias com one hot encoding
    brewmethod_encode = pd.get_dummies(selecao['BrewMethod' ])

    #Excluindo a coluna de texto 'BrewMethod'
    selecao.drop('BrewMethod', axis = 1, inplace = True)

    #Inserindo as variáveisIone hot encode novamente no dataset
    concatenado = pd.concat([selecao, brewmethod_encode], axis=1)

    concatenado['PitchRate' ].fillna(concatenado['PitchRate' ].mean(), inplace=True)
    concatenado. fillna(concatenado.median(), inplace=True)
    return concatenado

concatenado = processamento_dados()
y = concatenado['StyleID']
x = concatenado.drop('StyleID', axis=1)

import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

valores_grid = {'learning_rate': np.array([0.9,0.8,0.7,0.6,0.5,0.4])}

modelo = AdaBoostClassifier(n_estimators=500)

gridAdaBoost = GridSearchCV(estimator=modelo,param_grid=valores_grid,cv=3,n_jobs=-1)
gridAdaBoost.fit(x,y)

print("Melhor taxa de aprendizagem: ", gridAdaBoost.best_estimator_.learning_rate)
print('Acurácia: ', gridAdaBoost.best_score_)

# a performance anda não está boa. O ideal seria fazer mais rodadas, diminuindo o learning rate
