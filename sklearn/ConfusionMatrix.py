from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('display.max_columns', 30)
data = load_breast_cancer()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(y.value_counts()) # verificar se o dataset está desproporcional. Nesse caso, não está

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

X_treino, X_teste, Y_treino, Y_teste = train_test_split(x, y, test_size = 0.3, random_state = 9)

modelo = LogisticRegression(C=95, penalty='l2', max_iter=5000)
modelo.fit(X_treino,Y_treino)

resultado = modelo.score(X_teste,Y_teste)
print('Acurácia:', resultado)

predicao = modelo.predict(X_teste)
print(predicao)

from sklearn.metrics import confusion_matrix
matriz = confusion_matrix(Y_teste, predicao)
print(matriz)
