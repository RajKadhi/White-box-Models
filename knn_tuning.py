#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import PolynomialFeatures
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


# In[2]:


df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ';')
y = df.pop('quality')


# In[3]:


for i in df.columns:
    df[i] = df[i].fillna(np.mean(df[i]))
train, test, y_train, y_test = train_test_split(df, y, test_size = 0.2)


# In[4]:


lr = LogisticRegression()
lr.fit(train, y_train)
y_pred = lr.predict(test)
print('Accuracy score baseline:', accuracy_score(y_test, y_pred))


# In[5]:


def fit_predict(train, test, y_train, y_test, scaler, 
                n_neighbours, metric = 'manhattan', weights = 'uniform'):
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)        
    knn = KNeighborsClassifier(n_neighbors=n_neighbours, metric=metric, 
                               weights=weights, n_jobs = 4)
    knn.fit(train_scaled, y_train)
    y_pred = knn.predict(test_scaled)
    print(accuracy_score(y_test, y_pred))


# ### Neighbours tuning

# In[6]:


for k in range(1,12):
    print('Accuracy score on kNN using n_neighbours = {0}:'.format(2**k), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), 2**k)


# In[7]:


for k in np.logspace(2, 11, base = 2, num = 11, dtype=int).tolist():
    print('Accuracy score on kNN using n_neighbours = {0}:'.format(k), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), k)


# ### Metric tuning

# In[8]:


for metric in ['euclidean', 'cosine', 'manhattan', 'chebyshev']:
    print('Accuracy score on kNN using {} metric and {} neighbours:'.format(metric,k), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), 2, metric)


# ### Weighted kNN

# In[9]:


for weights in ['uniform', 'distance']:
    print('Accuracy score on kNN using weights = {0}:'.format(weights), end = ' ')
    fit_predict(train, test, y_train, y_test, StandardScaler(), 2, 'chebyshev', weights = weights)


# ### Engineering

# In[10]:


def create_poly(train,test,degree):
    poly = PolynomialFeatures(degree=degree)
    train_poly = poly.fit_transform(train)
    test_poly = poly.fit_transform(test)
    return train_poly,test_poly


# In[11]:


for degree in [1,2,3]:
    train_poly, test_poly = create_poly(train, test, degree)
    print('Polynomial degree',degree)
    fit_predict(train_poly, test_poly, y_train, y_test, StandardScaler(), 2, 'chebyshev', weights = 'distance')
    print(10*'-')
    
train_poly, test_poly = create_poly(train, test, 2) 


# In[12]:


def feat_eng(df):
    df['eng1'] = df['fixed acidity'] * df['pH']
    df['eng2'] = df['total sulfur dioxide'] / df['free sulfur dioxide']
    df['eng3'] = df['sulphates'] / df['chlorides']
    df['eng4'] = df['chlorides'] / df['sulphates']
    return df

train = feat_eng(train)
test = feat_eng(test)


# In[13]:


print('Accuracy score after engineering:', end = ' ')
fit_predict(train, test, y_train, y_test, StandardScaler(), 2, 'chebyshev', weights = 'distance')


# In[14]:


original_score = 0.514285714286
best_score = 0.670408163265
improvement = np.abs(np.round(100*(original_score - best_score)/original_score,2))
print('overall improvement is {} %'.format(improvement))


# In[ ]:





# In[ ]:





# In[ ]:




