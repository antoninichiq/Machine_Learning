from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('display.max_columns', 30)
data = load_breast_cancer()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler # função de normalização. Cada coluna tem uma característica diferente. Uma tem ordem de grandeza na ordem de 0.02 e outra na ordem das dezenas. 
from sklearn.model_selection import train_test_split

#Normalizando as variáveis preditoras
normalizador = MinMaxScaler(feature_range= (0, 1))
X_norm = normalizador.fit_transform(x)

#Seprando os dados entre treino e teste:
X_treino, X_teste, Y_treino, Y_teste = train_test_split(X_norm, y, test_size = 0.3, random_state=16)

#Criação do modelo:
modelo = KNeighborsClassifier(n_neighbors=5)
modelo.fit(X_treino, Y_treino)

#Score
resultado = modelo.score(X_teste, Y_teste)
print("Acurácia:", resultado)
