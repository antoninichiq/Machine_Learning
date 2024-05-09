from sklearn.feature_selection import SelectKBest #seleciona o melhor score
from sklearn.feature_selection import chi2

#Definindo variáveis preditoras e target (mesmo exemplo do caderno)
x = [[12,2,30],[15,11,6],[16,8,90],[5,3,20],[4,14,5],[2,5,70]]
y = [1,1,1,0,0,0]

#Selecionando duas variváveis com o maior chi-quadrado
algoritmo = SelectKBest(score_func=chi2,k=2)
dados_das_melhores_preditoras = algoritmo.fit_transform(x,y)

#Resultados
print("Score: ", algoritmo.scores_)
print("Resultado da transformação:\n", dados_das_melhores_preditoras)
