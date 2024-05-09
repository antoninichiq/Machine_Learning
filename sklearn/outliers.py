import pandas as pd
pd.set_option('display.max_columns',50)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (5)/traffic-collision-data-from-2010-to-present.csv")

area_encode = pd.get_dummies(arquivo['Area Name']) #Temos nomes ao inves de número nessa coluna, por isso fazemos o one hot encoding

concatenando = pd.concat([arquivo, area_encode], axis = 1)
concatenando.drop('Area Name', axis = 1, inplace = True)

import matplotlib.pyplot as plt
concatenando.boxplot(column = 'Census Tracts')
plt.show() # os valores acima dos 1500 são resíduos, discrepantes