import pandas as pd
pd.set_option('display.width', 320)
dados = pd.read_csv("C:/Users/anton/Downloads/archive (2)/diabetes.csv")
print(dados.corr(method = 'pearson')) #Pearson criou uma fórmula que mede a correlação (o quanto estão relacionadas) entre cada variável 

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
sns.heatmap(dados.corr()) #ver mapa de calor
plt.show()