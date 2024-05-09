import pandas as pd
pd.set_option('display.max_columns',23)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (6)/recipeData.csv", encoding='latin-1')
# Ao invés de completar o dataset simplesmente com a média ou mediana, utilizaremos um modelo para prever esses dados

arquivo = arquivo.loc[arquivo['StyleID'].isin([7,10,154,9,4,30,86,12,92,6,175,39])]

arquivo.drop('BeerID', axis = 1, inplace = True)
arquivo.drop('Name', axis = 1, inplace = True)
arquivo.drop('URL', axis = 1, inplace = True)
arquivo.drop('Style', axis = 1, inplace = True)
arquivo.drop('UserId', axis = 1, inplace = True)
arquivo.drop('PrimingMethod', axis = 1, inplace = True)
arquivo.drop('PrimingAmount', axis = 1, inplace = True)

# ----------------------------
#One hot Enconding
arquivo['SugarScale'] = arquivo['SugarScale'].replace("Specific Gravity",0)
arquivo['SugarScale'] = arquivo['SugarScale'].replace("Plato",1)

brewmethod_encode = pd.get_dummies(arquivo['BrewMethod'])
arquivo.drop('BrewMethod', axis = 1, inplace=True)
concatenando = pd.concat([arquivo, brewmethod_encode],axis=1)
# -----------------------------

print(concatenando.head(15))

faltantes = concatenando.isnull().sum()
faltantes_percentual = (concatenando.isnull().sum() / len(concatenando['StyleID'])) * 100
print(faltantes_percentual)

concatenando['BoilGravity'].fillna(concatenando['BoilGravity'].median(),inplace=True)

#Faremos PitchRate de target (y) e todas as outras de preditores (x)
x_treino = concatenando[concatenando['PitchRate'].notnull()] # pegando os valores que nao sao nulos
x_treino.drop('PitchRate', axis = 1, inplace=True) #PitchRate é a variavel target, entao precisamo excluir do dataset
y_treino = concatenando[concatenando['PitchRate'].notnull()]['PitchRate']
x_prever = concatenando[concatenando['PitchRate'].isnull()]
y_prever = concatenando[concatenando['PitchRate'].isnull()]['PitchRate']
x_prever.drop('PitchRate',axis=1,inplace=True)

x_treino.drop('MashThickness', axis=1,inplace=True)
x_treino.drop('PrimaryTemp', axis=1,inplace=True)
x_prever.drop('MashThickness', axis=1,inplace=True)
x_prever.drop('PrimaryTemp', axis=1,inplace=True)

from sklearn.tree import DecisionTreeRegressor
modelo = DecisionTreeRegressor()
modelo.fit(x_treino,y_treino)

y_prever=modelo.predict(x_prever) #completando os valores NaNs com os valores previstos

#colocando os valores previstos no dataset
concatenando.PitchRate[concatenando.PitchRate.isnull()] = y_prever

faltantes = concatenando.isnull().sum()
faltantes_percentual = (concatenando.isnull().sum() / len(concatenando['StyleID'])) * 100
print(faltantes_percentual)


pd.set_option('display.max_rows', None)
print(concatenando['PitchRate'].value_counts())
print(concatenando['PitchRate'].head())
