import pandas as pd
pd.set_option('display.max_columns',50)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (5)/traffic-collision-data-from-2010-to-present.csv")

# print(arquivo.shape)
# print(arquivo.dtypes)
# print(arquivo.head())
# faltantes = arquivo.isnull().sum()
# faltantes_percentual = (arquivo.isnull().sum() / len(arquivo['DR Number'])) * 100
# print(faltantes_percentual)

area_encode = pd.get_dummies(arquivo['Area Name']) #Temos nomes ao inves de número nessa coluna, por isso fazemos o one hot encoding
print(area_encode.head())

concatenando = pd.concat([arquivo, area_encode], axis = 1)
concatenando.drop('Area Name', axis = 1, inplace = True)
print(concatenando.head())