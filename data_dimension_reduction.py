#!/usr/bin/env python
# coding: utf-8

# # Data Dimensionality Reduction
# Comparar os 4 métodos de redução de dimensões (PCA, SVD, NMF e Autoencoder) em relação ao modelo sem nenhum tipo de redução.  
# Para tornar o exercício mais interessante, será avaliado não só os 4 algoritmos de redução de dimensão, mas também vários modelos de classificação.  
# O objetivo final é termos a melhor opção entre o modelo de classificação e de redução (se for o caso).  
# 
# ### Base de dados - SK-Learn Breast Cancer
# 
# Para o exercício será utilizado o dataset do sklearn **breast cancer** (UCI ML Breast Cancer) no que contém informações sobre câncer de mama e a classificação se é benigno ou maligno.  
# Maiores detalhes sobre o dataset disponível em  
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html#sklearn.datasets.load_breast_cancer
# 
# 
# ***Fonte original do dataset***  
# The copy of UCI ML Breast Cancer Wisconsin (Diagnostic) dataset is downloaded from:  
# https://goo.gl/U2Uwz2
# 
# 
# ### Instalando as bibliotecas necessárias
# Esta etapa pode ser pulada se o ambiente já estiver configurado com as bibliotecas  
# *A instalação do  tensorflow foi recomendada pelo site do KERAS embora não utilizemos no exercício*

# !pip install pandas
# !pip install numpy
# !pip install matplotlib
# !pip install seaborn
# !pip install -U scikit-learn
# !pip install tensorflow #keras
# !pip install keras #keras

# ## Importando o Dataset
# O conjunto de dados utilizado contém possui em seu target uma classificação binária do câncer em 'malignant', 'benign' (maligno ou benigno).  
# O objetivo, é criar um classificador que ao avaliar suas 30 features possa determinar o tipo do cancer.
# 

# In[1]:


#importando o conjunto de dados
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()


# ### Exibindo as Features e targets

# In[2]:



print('Target:', data.target_names, '\n')
print('Features:',data.feature_names)


# ### Importando as demais bibliotecas utilizadas

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split
from sklearn import model_selection 
#plt.style.use('ggplot')

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")


# ### Separando os dados em treino e teste
# 
# Para o exemplo utilizaremos uma base de tete de 30% do conjunto de dados.

# In[4]:


random_state = 42
#Carregndo novamente já seprando os dados
X, y = load_breast_cancer(return_X_y=True)

#separando em treino e teste com 40% para o teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)


# ### Definindo os modelos que serão utilizados 
# 
# Em princípio faremos testes com 5 tipos declassificadores

# In[5]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

#import warnings
#from sklearn.exceptions import ConvergenceWarning
#warnings.filterwarnings("ignore",  category = ConvergenceWarning)

def rodar_modelo(models):
    '''Executa os modelos definidos (parametros default) e retorna o modelo com melhor performance'''
    results = []
    names = []
    scoring = 'accuracy'
    print('Resultado da avaiação')
    for name, model in models:
        kfold = model_selection.KFold(n_splits=5, random_state=random_state, shuffle=True)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results.mean())
        names.append(name)
        msg = "  %s: media %f (dp %f - max %f)" % (name, cv_results.mean(), cv_results.std(), cv_results.max())
        print(msg)
    pos_melhor_modelo = results.index(max(results))
    return models[pos_melhor_modelo]


models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('DTC', DecisionTreeClassifier()))
models.append(('GB', GaussianNB()))
models.append(('SVC', SVC()))
models.append(('RF', RandomForestClassifier()))

melhor_modelo = rodar_modelo(models)
print('')
print('Melhor modelo:', melhor_modelo[1])


# ### Aprimorando o modelo selecionado 'Random Forest Classifier'
# 
# Utilizar o GridSearch para testar um conjunto de hipoteses de parametros. O objetivo é descobrir a melhor combinação entre eles e maximizar o resultado da classificação.

# In[6]:



params= {
    'random_state': [None, 0, 1, 2, 3, 42],
    'max_depth': [None, 1, 2, 3, 4, 15, 20],
    'n_estimators': [5, 10, 100, 150, 200], 
    'max_features': ['auto', 'sqrt', 'log2', 1, 2, 3],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 3, 4, 5]
}

gsc = GridSearchCV(estimator=RandomForestClassifier(),param_grid=params ,cv=5, scoring='accuracy', verbose=0, n_jobs=-1)

grid_result = gsc.fit(X_train, y_train)
best_params = grid_result.best_params_

classifier = RandomForestClassifier(max_depth=best_params["max_depth"], 
                                    random_state=best_params["random_state"],
                                    n_estimators = best_params["n_estimators"], 
                                    max_features = best_params["max_features"],
                                    criterion=best_params["criterion"], 
                                    min_samples_split=best_params["min_samples_split"], 
                                    min_samples_leaf = best_params["min_samples_leaf"])
print("RandomForestClassifier GIRD")
for k, v in best_params.items():
    print(' ', k, ': ',  v,)


# In[7]:


results = []
model = classifier.fit(X_train, y_train)
y_pred = model.predict(X_test) # avaliando o melhor estimador


print("")
print(classification_report(y_test, y_pred, target_names=['Maligno', 'Benigno']))
report = classification_report(y_test, y_pred, target_names=['Maligno', 'Benigno'], output_dict=True)
results.append(('Modelo', report))
plot_confusion_matrix(classifier, X_test, y_test, cmap="Blues", display_labels=['Maligno', 'Benigno'])
plt.show()


# ### Avaliando as features do modelo quanto a sua importancia

# In[8]:


from matplotlib.pyplot import figure
figure(figsize=(10, 8), dpi=80)

def avaliar_features(melhor_modelo):
    model = melhor_modelo[1]
    model.fit(X_train, y_train)
    importance = model.feature_importances_
    colunas = data.feature_names
    values = []
    # summarize feature importance
    for i,v in enumerate(importance):
        print('%s: %.5f' % (colunas[i],v))
        values.append(v)

    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)
    std_var = np.std(values)
    media = np.mean(values)

    
    plt.axhline(y= media, linestyle="--", color='k')
    plt.annotate('%s: %.2f' % ('Média',media), xy=(30, media), xycoords='data', xytext=(20, 4), textcoords='offset points', fontsize=12, color='k')
    
    plt.axhline(y= std_var, linestyle="--", color='k')
    plt.annotate('%s: %.2f' % ('Desvio padrão',std_var), xy=(30, std_var), xycoords='data', xytext=(20, 4), textcoords='offset points', fontsize=12, color='k')
   
    plt.title("Importância de cada Feature para o modelo")
    plt.show()

avaliar_features(melhor_modelo)


# ## Aplicando os algoritmos de redução de dimensões para comparar os resultados
# 
# Com base no gráfico, pecebemos que cada uma das features possui um grau diferente de relevancia para o algoritmo. Avaliar a possibilidade de diminuir a quantidade de features economizaria tempo computacional considerável.

# ### PCA - Principal Component Analysis

# In[9]:


#Aplicar a redução de dimensão em X e rodar o processo novamente com os melhores parametros
from sklearn.decomposition import PCA as sklearnPCA

#escalando dados
standard_scaler = StandardScaler()
X_rescaled  = standard_scaler.fit_transform(X)
pct_components = 0.95

pca = sklearnPCA(n_components=pct_components)
X_pca = pca.fit_transform(X_rescaled)
X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split(X_pca, y, test_size=0.4, random_state=random_state)

#Aplicar o novo X no modelo já treinado
model = classifier.fit(X_train_pca, y_train_pca)
y_pred_pca = model.predict(X_test_pca) # avaliando o melhor estimador

report = classification_report(y_test_pca, y_pred_pca, target_names=['Maligno', 'Benigno'], output_dict=True)
results.append(('PCA', report))
print(classification_report(y_test_pca, y_pred_pca, target_names=['Maligno', 'Benigno']))
plot_confusion_matrix(classifier, X_test_pca, y_test_pca, cmap="Blues", display_labels=['Maligno', 'Benigno'])
plt.show()


# ### Exibindo o gráfico de componentes necessários com base em uma variancia de 0.95%
# Formato de dados antes e depois do PCA

# In[10]:


plt.rcParams["figure.figsize"] = (12,6)

fig, ax = plt.subplots()
xi = np.arange(1, X_pca.shape[1]+1, step=1)
yi = np.cumsum(pca.explained_variance_ratio_)

plt.ylim(0.0,1.1)
plt.plot(xi, yi, marker='o', linestyle='--', color='b')

plt.xlabel('Numero de componentes')
plt.xticks(np.arange(0, X_pca.shape[1]+2, step=1)) #mudando a escala do eixo só pra ficar mais bacana
plt.ylabel('Variancia acumulada (%)')
plt.title('Número de componentes necessários para variancia')

plt.axhline(y=pct_components, color='r', linestyle='-')
plt.text(0.5, 0.90, 'Limite de 95%', color = 'red', fontsize=16)

ax.grid(axis='x')
plt.show()


# ### SVD - Single Value Decomposition

# In[19]:


U, S, V = np.linalg.svd(X_rescaled)
k = 7
df = pd.DataFrame({'x_values': list(range(len(S))), 'y_values': S })
plt.rcParams["figure.figsize"] = (12,6)
plt.plot('x_values', 'y_values', data=df, color='red')
plt.scatter(k, S[k], color='blue')
plt.title("Avaliação da Curva de Erro x Componentes")
plt.xlabel("Numero de Componentes")
plt.ylabel("Erro")
plt.show()


# ### Avaliando a quantidade de componentes
# Com a avaliação do gráfico, consideraremos 7 componentes

# In[12]:


from sklearn.decomposition import TruncatedSVD as sklearnSVD

#normalizando os dados
standard_scaler = StandardScaler()
X_rescaled  = standard_scaler.fit_transform(X)

svd = sklearnSVD(n_components=7)
X_svd = svd.fit_transform(X_rescaled)

X_train_svd, X_test_svd, y_train_svd, y_test_svd = train_test_split(X_svd, y, test_size=0.4, random_state=random_state)

#Aplicar o novo X no modelo já treinado
model = classifier.fit(X_train_svd, y_train_svd)
y_pred_svd = model.predict(X_test_svd) # avaliando o melhor estimador

report = classification_report(y_test_svd, y_pred_svd, target_names=['Maligno', 'Benigno'], output_dict=True)
results.append(('SVD', report))
print(classification_report(y_test_svd, y_pred_svd, target_names=['Maligno', 'Benigno']))
plot_confusion_matrix(classifier, X_test_svd, y_test_svd, cmap="Blues", display_labels=['Maligno', 'Benigno'])
plt.show()


# ### NMF - Non-negative Matrix Factorization

# In[13]:


from sklearn.decomposition import NMF as sklearnNMF

nmf = sklearnNMF(n_components=10)

#escalando dados
minmax_scaler = MinMaxScaler()
X_rescaled  = minmax_scaler.fit_transform(X)
X_train_nmf, X_test_nmf, y_train_nmf, y_test_nmf = train_test_split(X_rescaled, y, test_size=0.4, random_state=random_state)

#Aplicar o novo X no modelo já treinado
model = classifier.fit(X_train_nmf, y_train_nmf)
y_pred_nmf = model.predict(X_test_nmf) # avaliando o melhor estimador

print(classification_report(y_test_nmf, y_pred_nmf, target_names=['Maligno', 'Benigno']))
report = classification_report(y_test_nmf, y_pred_nmf, target_names=['Maligno', 'Benigno'], output_dict=True)
results.append(('NMF', report))
plot_confusion_matrix(classifier, X_test_nmf, y_test_nmf, cmap="Blues", display_labels=['Maligno', 'Benigno'])
plt.show()


# ### Autoencoder

# In[14]:


## Import do keras
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras import regularizers

#normalizando e preparando os dados
standard_scaler = StandardScaler()
X_rescaled  = standard_scaler.fit_transform(X)

X_train_ae, X_test_ae, y_train_ae, y_test_ae = train_test_split(X_rescaled, y, test_size=0.4, random_state=random_state)


# seguindo a recomendação, testando o modelo com vários parametros similar ao gridsearv do sklearn avaliando a melhor confirugação
L1_val = [1e-8, 1e-7, 1e-6]
L2_val = [1e-8, 1e-7, 1e-6]

#numero de dimensoes que desejamos reduzir
encoding_dim = 10

## Valores de erro
Grid_Val = []

shape_in = X_train_ae.shape[1]

## Loop nos regularizadores
for L1 in L1_val:
    for L2 in L2_val:
        input_df = Input(shape=(shape_in,))
        encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2))(input_df)
        decoded = Dense(shape_in, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2))(encoded)

        # encoder
        autoencoder = Model(input_df, decoded)

        # intermediate result
        encoder = Model(input_df, encoded)

        autoencoder.compile(optimizer='adadelta', loss='mean_squared_error')

        autoencoder_history = autoencoder.fit(X_train_ae, X_train_ae,
                                             epochs=50,
                                             batch_size=128,
                                             shuffle=True,
                                             validation_data=(X_test_ae, X_test_ae))

        ## Adiciona o erro de teste na lista
        Grid_Val.append([L1,L2,autoencoder_history.history["val_loss"][-1]])
        
        ## Print do andamento
        print("Complete L1:", L1, " L2:", L2)

## Transforma em array
Grid_Val = np.array(Grid_Val)


# ### Verificando a melhor combinação
# Após rodar a rede com vários parametros, construímos uma nova rede com os melhores selecionados

# In[15]:


## Selecao do melhor modelo
print(Grid_Val[Grid_Val[:,2] == Grid_Val[:,2].min(),:])
L1 = Grid_Val[Grid_Val[:,2] == Grid_Val[:,2].min(),0][0]
L2 = Grid_Val[Grid_Val[:,2] == Grid_Val[:,2].min(),1][0]
print("L1:",L1,"L2:",L2)


# In[16]:


#Construindo 
input_df = Input(shape=(shape_in,))
encoded = Dense(encoding_dim, activation='relu', kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2))(input_df)
decoded = Dense(shape_in, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=L1, l2=L2))(encoded)

# encoder
autoencoder = Model(input_df, decoded)

# intermediate result
encoder = Model(input_df, encoded)

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

autoencoder_history = autoencoder.fit(X_train_ae, X_train_ae,
                                     epochs=50,
                                     batch_size=128,
                                     shuffle=True,
                                     validation_data=(X_test_ae, X_test_ae))


#Montando os dados novos para usarmos no modelo
encoded_X_train = encoder.predict(X_train_ae)
encoded_X_test = encoder.predict(X_test_ae)


# ### Rodando o modelo com o novo cenário

# In[17]:



#Aplicar o novo X no modelo já treinado
model = classifier.fit(encoded_X_train, y_train_ae)
y_pred_ae = model.predict(encoded_X_test) # avaliando o melhor estimador

print(classification_report(y_test_ae, y_pred_ae, target_names=['Maligno', 'Benigno']))
report = classification_report(y_test_ae, y_pred_ae, target_names=['Maligno', 'Benigno'], output_dict=True)
results.append(('AE', report))
plot_confusion_matrix(classifier, encoded_X_test, y_test_ae, cmap="Blues", display_labels=['Maligno', 'Benigno'])
plt.show()


# ### Comparando os resultados
# O gráfico abaixo compara os resultados de acurácia obtidos

# In[18]:


import numpy as np
import matplotlib.pyplot as plt

#report
#macro_precision =  results[0]['macro avg']['precision'] 
#macro_recall = report['macro avg']['recall']    
#macro_f1 = report['macro avg']['f1-score']
#accuracy = report['accuracy']

import numpy as np
import matplotlib.pyplot as plt

barWidth = 0.20

# data to plot
acc = [results[0][1]['accuracy'], results[1][1]['accuracy'], results[2][1]['accuracy'], results[3][1]['accuracy'], 0.7]
#f1 = [85, 62, 54, 20, 40]
#precision = [85, 62, 54, 20, 30]
#recall = [85, 62, 54, 20, 30]


# Set position of bar on X axis
r1 = np.arange(len(acc))
#r2 = [x + barWidth for x in r1]
#r3 = [x + barWidth for x in r2]
#r4 = [x + barWidth for x in r3]

fig = plt.figure(figsize=(15,10))
plt.title('Comparativo dos resultados', fontsize=16, fontweight='bold')

#make the plot
plt.bar(r1, acc, color='#7972fc', width=barWidth, edgecolor='white', label='Accuracy')
#plt.bar(r2, f1, color='#fc3a51', width=barWidth, edgecolor='white', label='F1')
#plt.bar(r3, precision, color='y', width=barWidth, edgecolor='white', label='Precision')
#plt.bar(r4, recall, color='#4aba50', width=barWidth, edgecolor='white', label='Recall')

# Add xticks on the middle of the group bars
plt.xlabel('Algoritmos', fontweight='bold')
plt.xticks([r  for r in range(len(acc))], ['Nenhum', 'PCA', 'SVD', 'NMF', 'AE'])

for label_x,label_y in zip(r1,acc):
    label = "{:.2f}".format(label_y)
    plt.annotate(label, 
                 (label_x,label_y), 
                 textcoords="offset points", 
                 xytext=(0,10), 
                 ha='center')
plt.legend()
plt.show()


# ### Conclusão
# Com base nas observações o melhor resultado de acurácia (96%) foi com aplicação do PCA, ou seja, com a redução de dimensões é possível reduzirmos o consumo computacional e aumentar o ganho dos algoritmos.
# 
# 
# ### Referências
# Sites da Internet  
# https://blog.paperspace.com/dimension-reduction-with-autoencoders/  
# https://machinelearningmastery.com/binary-classification-tutorial-with-the-keras-deep-learning-library/  
# 
# 
# Artigos do Medium  
# https://python.plainenglish.io/non-negative-matrix-factorization-for-dimensionality-reduction-predictive-hacks-1ed9e91154c  
# https://medium.com/@rinu.gour123/dimensionality-reduction-in-machine-learning-dad03dd46a9e  
# https://medium.com/analytics-vidhya/a-complete-guide-on-dimensionality-reduction-62d9698013d2  
# https://towardsdatascience.com/tagged/dimensionality-reduction  
# 
# Livros  
# Python Data Science Essentials - Third Edition now with O’Reilly online learning.  
# Python Data Science Handbook by Jake VanderPlas; Jupyter notebooks are available on [GitHub](https://github.com/jakevdp/PythonDataScienceHandbook)  

# In[ ]:




