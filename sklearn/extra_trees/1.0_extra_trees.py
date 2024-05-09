from sklearn.datasets import load_breast_cancer
import pandas as pd
pd.set_option('display.max_columns',30)
dados = load_breast_cancer()
x = pd.DataFrame(dados.data,columns=[dados.feature_names])
y = pd.Series(dados.target)

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

modelo = ExtraTreesClassifier(n_estimators=50)
skfold = StratifiedKFold(n_splits=3)
resultado = cross_val_score(modelo,x,y,cv=skfold)
print(resultado.mean())
