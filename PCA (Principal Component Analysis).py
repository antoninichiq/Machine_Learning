from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
x = pd.DataFrame(iris.data, columns = [iris.feature_names])
y = pd.Series(iris.target)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

normalizador = MinMaxScaler(feature_range=(0,1))
x_norma = normalizador.fit_transform(x)

pca = PCA(n_components=2) # PC1 e PC2
x_pca = pca.fit_transform(x_norma)

x_treino, x_teste, y_treino,y_teste = train_test_split(x_pca, y, test_size=0.3, random_state=14)

modelo = KNeighborsClassifier()
modelo.fit(x_treino,y_treino)

score = modelo.score(x_teste,y_teste)
print("Acurácia: ", score)

print("Variância explicada dos componentes:",pca.explained_variance_ratio_)