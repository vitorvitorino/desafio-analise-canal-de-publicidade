# Objetivo

Este notebook é a resolução do desafio principal para a conclusão do módulo de regressão com Python, o desafio consistiu em explorar dados de publicidade para entender a relação entre os investimentos em diferentes canais de marketing (facebook, youtube, newspaper) e as vendas (sales). Por meio de técnicas de análise descritiva, exploratória e modelagem preditiva, busquei identificar os canais que mais influenciam as vendas e propor estratégias para otimizar a alocação de recursos.

# Diagnóstico Inicial:
A análise estatística (.describe()) revela um desvio padrão considerável nas colunas 'youtube', 'facebook' e 'newspaper'. Isso pode indicar tanto uma alocação de recursos variável nas estratégias de marketing quanto a influência de fatores sazonais e o impacto de diferentes campanhas. A coluna 'sales', no entanto, demonstra um desvio padrão inferior, sugerindo uma consistência maior nos resultados de vendas dentro do dataset.

![image](https://github.com/user-attachments/assets/bab98203-a310-4a36-92c4-75332faab542)

```python
# Importando bibliotecas e inspecionando o dataset
import pandas as pd
dataset = pd.read_csv('/content/MKT.csv')
dataset.info()
dataset.describe()
```

# Análise Exploratória:

# Histogramas

A análise dos histogramas revela que os investimentos nos canais 'facebook' e 'newspaper' apresentam uma concentração significativa no início do gráfico. Este padrão sugere que a maior parte dos investimentos direcionados a essas plataformas corresponde a valores relativamente mais baixos.

Observa-se uma tendência semelhante no histograma da coluna 'sales', onde o pico principal e os demais picos menores se localizam no início do eixo X. Isso indica que, com frequência, os valores de vendas tendem a ser menores dentro do conjunto de dados.

Complementando essa observação, a coluna 'sales' exibe os menores valores em todo o dataset, com um valor máximo registrado de 32.

Em contraste, o histograma da coluna 'youtube' se estende por toda a extensão do eixo X, demonstrando que os investimentos neste canal abrangem uma gama maior de valores, incluindo montantes mais elevados em comparação com 'facebook' e 'newspaper'. A presença de múltiplos picos ao longo do eixo X sugere uma variedade nos valores investidos em 'youtube'.

![image](https://github.com/user-attachments/assets/9609ccdd-4d1e-42a9-bf2d-1f1c87760116)

```python
# Importando Seaborn e Matplotlib para plotar o histograma e analisar a distribuição dos valores:

import seaborn as sns
import matplotlib.pyplot as plt

for d in dataset.columns:
  plt.figure()
  sns.histplot(data = dataset[d], bins = 20)
  plt.title(f"Histograma de {d}")

# Plotando boxplot para comparar a distribuição e corroborar o histograma:

plt.figure(figsize = (8, 6))
sns.boxplot(data = dataset);
```
# Boxplot 

O boxplot de 'youtube' revela que os investimentos tendem a ser mais altos (mediana superior) e apresentam uma variabilidade considerável (tamanho da caixa), com a ocorrência de valores tanto significativamente altos quanto mais baixos (bigodes estendidos).

O boxplot de 'facebook' demonstra uma menor variação nos valores de investimento (caixa menor) e uma tendência a valores mais baixos (mediana inferior). A curta extensão dos bigodes sugere que os investimentos raramente fogem muito dessa faixa.

Em 'newspaper', a mediana também indica investimentos mais modestos, semelhante a 'facebook'. No entanto, a caixa ligeiramente maior aponta para uma variação um pouco maior nos valores, e o bigode superior alongado com outliers demonstra a ocorrência de alguns investimentos atipicamente altos.

O boxplot de 'sales' destaca-se pela baixa variabilidade nos valores (caixa pequena e bigodes curtos) e indica que os valores de venda tendem a ser mais baixos (mediana inferior).

![image](https://github.com/user-attachments/assets/b8f2d901-d963-44f4-bfc9-e790954aeff4)

```python
# Plotando boxplot para comparar a distribuição e corroborar o histograma:

plt.figure(figsize = (8, 6))
sns.boxplot(data = dataset);
```
# Heatmap e Pairplots

A análise conjunta do heatmap de correlação e dos pairplots revela uma forte correlação linear positiva entre os investimentos em 'youtube' e as 'sales'. Essa relação é evidenciada pelo alto coeficiente de correlação de 0.78 e pela clara tendência ascendente observada no gráfico de dispersão (scatterplot), onde os pontos se agrupam em uma direção que vai do canto inferior esquerdo ao superior direito.

Da mesma forma, identifica-se uma correlação positiva entre os investimentos em 'facebook' e 'sales', embora de intensidade um pouco menor. O coeficiente de correlação de 0.6 indica uma correlação moderada, e o scatterplot correspondente apresenta uma dispersão ligeiramente maior dos pontos, mantendo, contudo, uma tendência de crescimento.

Em contraste, a correlação entre os investimentos em 'newspaper' e 'sales' é fraca, com um coeficiente de correlação de apenas 0.25. Essa baixa correlação se reflete no scatterplot, que exibe uma distribuição de pontos aleatória, sem uma direção predominante clara. Essa observação é particularmente relevante considerando que os investimentos em 'newspaper' são, em alguns casos, maiores do que os investimentos em 'facebook', levantando questionamentos sobre a eficiência da alocação de recursos nesse canal.

![image](https://github.com/user-attachments/assets/bf8b0842-1cc5-464d-b757-f9a9cd8f0593)
![image](https://github.com/user-attachments/assets/d57fab0f-cd51-4da2-b08b-90b5e3cd0d26)

```python
# Heatmap evidenciando gráficamente a correlação entre as colunas:

plt.figure(figsize = (8, 6))
sns.heatmap(data = datasetCorr, annot = True, cmap = 'coolwarm')
plt.title('Matriz de correlação');

# Pairplot isolado da coluna 'sales' no eixo y:

sns.pairplot(data = dataset, x_vars = ['youtube', 'facebook', 'newspaper', 'sales'], y_vars=['sales']);
```


# Modelagem Preditiva:

Para entender melhor como os investimentos impactam as vendas, foi aplicado um modelo de Regressão Linear. A variável dependente (sales) foi prevista com base nos valores de facebook, youtube e newspaper.
Resultados do Modelo:
facebook apresentou o maior coeficiente positivo, indicando que é o canal mais eficaz para impulsionar as vendas.
youtube também contribuiu positivamente, mas de forma menos significativa.
newspaper, por outro lado, apresentou coeficientes muito baixos, sugerindo que os investimentos nesse canal têm pouco ou nenhum impacto sobre as vendas.
A métrica de R² demonstrou que o modelo consegue explicar bem a variabilidade nas vendas, confirmando que os investimentos nos canais analisados são bons preditores do desempenho.

```python
# Importando a biblioteca Scikitlearn para criar o modelo:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Definindo as variaveis indepedendentes e a variável dependente:

x = dataset[['youtube', 'facebook', 'newspaper']]
y = dataset['sales']

# Dividindo os dados em test e train:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 78)

# Criando o modelo e treinando:

ml = LinearRegression()
ml.fit(x_train, y_train)

# Testando o modelo e avaliando o desempenho:

y_pred = ml.predict(x_test)

# R²

r2 = r2_score(y_test, y_pred)
print(f'Valor de r²: {r2:.2f}')

# MSE e RMSE (Mean squared error e Root mean squared error) para aprofundar a avaliação de desempenho:"

mse = mean_squared_error(y_test, y_pred)
print(f'Valor de MSE: {mse:.2f}')

rmse = np.sqrt(mse)
print(f'Valor de RMSE: {rmse:.2f}')

# Calculo dos coefficients:
coefficients = ml.coef_
print(coefficients)
```
# Modelagem

Para uma avaliação mais completa do modelo, além do R², foram incorporadas as métricas de Erro Quadrático Médio (MSE), Raiz do Erro Quadrático Médio (RMSE) e a análise dos Coeficientes. Essas métricas fornecem perspectivas adicionais sobre a capacidade preditiva do modelo.

O valor de R² obtido foi de 0.89, o que indica que o modelo explica aproximadamente 89% da variância na variável de vendas. Este resultado sugere um bom poder explicativo do modelo em relação aos dados observados.

O Erro Quadrático Médio (MSE) resultante foi de 3.70. A Raiz do Erro Quadrático Médio (RMSE), calculada como a raiz quadrada do MSE, foi de 1.92. O RMSE é uma métrica útil para quantificar o erro médio das previsões do modelo na unidade original da variável dependente. Neste caso, um RMSE de 1.92 significa que, em média, as previsões do modelo desviam do valor real de vendas em cerca de 1.92 unidades. Este valor considerado baixo é um bom indicador da precisão do modelo.

Os coeficientes estimados pelo modelo para cada canal foram: Youtube = 0.045, Facebook = 0.179 e Newspaper = 0.008. Esses coeficientes podem ser interpretados como o impacto médio de um aumento de uma unidade no investimento de cada canal sobre as vendas. Por exemplo, um coeficiente de 0.045 para Youtube significa que, para cada unidade adicional investida neste canal, espera-se um aumento médio de 0.045 unidades nas vendas.

A comparação dos coeficientes revela que o canal 'facebook' apresenta o maior coeficiente (0.179), sugerindo o maior retorno em vendas por unidade de investimento, seguido por 'youtube' (0.045) e, com o menor impacto, 'newspaper' (0.008).

```python
# Importando a biblioteca Scikitlearn para criar o modelo:

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Definindo as variaveis indepedendentes e a variável dependente:

x = dataset[['youtube', 'facebook', 'newspaper']]
y = dataset['sales']

# Dividindo os dados em test e train:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, random_state = 78)

# Criando o modelo e treinando:

ml = LinearRegression()
ml.fit(x_train, y_train)

# Testando o modelo e avaliando o desempenho:

y_pred = ml.predict(x_test)

# R²

r2 = r2_score(y_test, y_pred)
print(f'Valor de r²: {r2:.2f}')

# MSE e RMSE (Mean squared error e Root mean squared error) para aprofundar a avaliação de desempenho:

mse = mean_squared_error(y_test, y_pred)
print(f'Valor de MSE: {mse:.2f}')

rmse = np.sqrt(mse)
print(f'Valor de RMSE: {rmse:.2f}')

# Definindo coefficients:

coefficients = ml.coef_
print(coefficients)
```
# Conclusões finais

![image](https://github.com/user-attachments/assets/17972e9e-3309-48e8-9b17-cff6df758ceb)

```python
# Calculando uma predição com valores simulados:

valores = pd.DataFrame({'youtube': [100],
                        'facebook': [20],
                        'newspaper': [50]})

previsao = ml.predict(valores)

print(f"Valor de vendas: {previsao}")

# Calculando uma segunda predição com maior investimento em Facebook:

valores = pd.DataFrame({'youtube': [50],
                        'facebook': [100],
                        'newspaper': [20]})

previsao = ml.predict(valores)

print(f"Valor de vendas: {previsao}")

# Por fim, realizando uma predição com maior investimento em newspaper:

valores = pd.DataFrame({'youtube': [20],
                        'facebook': [50],
                        'newspaper': [100]})

previsao = ml.predict(valores)

print(f"Valor de vendas: {previsao}")

# Gráfico de dispersão (scatterplot) evidenciando o desempenho do modelo:

plt.figure(figsize=(8, 6))

sns.scatterplot(x=y_test, y=y_pred)

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

plt.xlabel('Valores reais')
plt.ylabel('Valores preditos');
```

Para avaliar comparativamente o impacto de diferentes estratégias de investimento em publicidade, foram conduzidas três simulações de previsão utilizando o modelo treinado. Cada simulação representou uma alocação distinta de recursos entre os canais 'youtube', 'facebook' e 'newspaper'.

O primeiro cenário simulou uma distribuição de investimento semelhante à observada nos dados reais, com 'youtube' recebendo a maior parte, seguido por 'newspaper' e, por fim, 'facebook'. Para esta configuração, o modelo previu um valor de vendas ('sales') de 12.16.

O segundo cenário foi concebido para explorar o potencial do canal 'facebook', que demonstrou a maior correlação com vendas na análise exploratória e apresentou o maior coeficiente no modelo de regressão. Assim, 'facebook' recebeu o maior investimento simulado, seguido por 'youtube' e 'newspaper'. O modelo indicou um valor de 'sales' de 24.01 para este cenário, um aumento expressivo em comparação com a simulação anterior.

O terceiro cenário focou em quantificar o impacto do canal 'newspaper'. Nesta simulação, 'newspaper' recebeu o maior investimento, seguido por 'facebook' e 'youtube'. A previsão resultante do modelo foi de 14.35 para 'sales', um valor superior ao cenário inicial, mas ainda abaixo daquele onde 'facebook' recebeu maior atenção.

Dado o R² do modelo (0.89), as previsões realizadas possuem uma boa margem de confiança. A análise dos resultados das simulações sugere fortemente que priorizar o investimento no canal 'facebook' pode levar a um aumento significativo nas vendas previstas. Em contrapartida, investir mais em 'newspaper' parece ter um impacto menos positivo. A conclusão é que um rebalanceamento do orçamento de marketing, com redução da alocação para 'newspaper' e aumento para 'facebook', é uma estratégia promissora, sendo recomendável uma análise mais aprofundada para determinar a alocação ideal entre 'facebook' e 'youtube'.
