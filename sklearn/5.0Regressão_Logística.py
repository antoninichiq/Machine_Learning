import pandas as pd
pd.set_option('display.max_columns',64)
pd.set_option('display.max_rows',64)
arquivo = pd.read_csv("C:/Users/anton/Downloads/archive (4)/Data_train_reduced.csv")

def processamento(arquivo,a=False): #No curso ele nao explicou o que é regressão logística
    #print(arquivo.head())
    #print(arquivo.shape)
    #print(arquivo.dtypes)
    if a:
        faltantes = arquivo.isnull().sum()
        faltantes_percentual = (faltantes / len(arquivo['Product'])) * 100
        print(faltantes_percentual)

    arquivo.drop('q8.20', axis=1, inplace=True)
    arquivo.drop('q8.18', axis=1, inplace=True)
    arquivo.drop('q8.17', axis=1, inplace=True)
    arquivo.drop('q8.10', axis=1, inplace=True)
    arquivo.drop('q8.9', axis=1, inplace=True)
    arquivo.drop('q8.8', axis=1, inplace=True)
    arquivo.drop('q8.2', axis=1, inplace=True)
    arquivo.drop('Respondent.ID', axis=1, inplace=True)
    arquivo.drop('Product', axis=1, inplace=True)
    arquivo.drop('q1_1.personal.opinion.of.this.Deodorant', axis=1, inplace=True)

    arquivo['q8.12'].fillna(arquivo['q8.12'].median(), inplace=True)
    arquivo['q8.7'].fillna(arquivo['q8.7'].median(), inplace=True)
    
    return arquivo

arquivo = processamento(arquivo)

y = arquivo['Instant.Liking']
x = arquivo.drop('Instant.Liking', axis = 1)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

#Separando os dados em folds:
stratifiedkfold = StratifiedKFold(n_splits=5)

#Criando o modelo:
modelo = LogisticRegression(max_iter=1000)
resultado = cross_val_score(modelo,x,y,cv=stratifiedkfold)

#Imprimindo a acurácia
print(resultado.mean())