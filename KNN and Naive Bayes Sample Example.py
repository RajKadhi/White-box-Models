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


# In[5]:


data = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ';')


# In[6]:


data.head()


# In[7]:


data.quality.unique()


# In[8]:


data.isna().sum()


# In[9]:


data.describe()


# In[10]:


data.info()


# In[14]:


data.shape


# In[15]:


data[data.duplicated()].shape


# In[17]:


data.drop_duplicates(inplace= True)


# In[18]:


data.shape


# In[19]:


data.corr()


# In[20]:


remove_column = ["citric acid", "sulphates","free sulfur dioxide" ]


# In[22]:


data.drop(columns=remove_column, inplace = True)


# In[ ]:


#data["quality"] = data["quality"].astype("category")


# In[25]:


X = data.drop(columns = "quality")


# In[26]:


y = data["quality"]


# In[27]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 1)


# In[28]:


knn = KNeighborsClassifier()


# In[29]:


knn.fit(train_X,train_y)


# In[31]:


train_pred = knn.predict(train_X)


# In[32]:


from sklearn.metrics import accuracy_score


# In[33]:


accuracy_score(train_pred,train_y)


# In[34]:


test_pred = knn.predict(test_X)


# In[35]:


accuracy_score(test_pred,test_y)


# In[64]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[65]:


scale = StandardScaler()
norm  = MinMaxScaler()


# In[66]:


X_scaled = scale.fit_transform(X)
X_normalised = norm.fit_transform(X)


# In[41]:


def model_fit(model, X, y):
    
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 1)
    model.fit(train_X,train_y)
    train_pred = model.predict(train_X)
    print("Train Accuracy :",accuracy_score(train_pred,train_y))
    test_pred = model.predict(test_X)
    print("Test Accuracy :", accuracy_score(test_pred,test_y))


# In[42]:


model_fit(knn, X, y)


# In[62]:


knn = KNeighborsClassifier(n_neighbors=57)


# In[63]:


model_fit(knn, X, y)


# In[43]:


model_fit(knn, X_scaled, y)


# In[67]:


model_fit(knn, X_normalised, y)


# In[71]:


knn = KNeighborsClassifier(n_neighbors=30)


# In[72]:


model_fit(knn, X_normalised, y)


# In[68]:



def model_fit_k_optimizer(k, X, y):
    model = KNeighborsClassifier(n_neighbors=k)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 1)
    model.fit(train_X,train_y)
    train_pred = model.predict(train_X)
    print("Train Accuracy :",accuracy_score(train_pred,train_y))
    train_acc = accuracy_score(train_pred,train_y)
    test_pred = model.predict(test_X)
    print("Test Accuracy :", accuracy_score(test_pred,test_y))
    test_acc = accuracy_score(test_pred,test_y)
    return train_acc, test_acc


# In[48]:


train_acc_list = []
test_acc_list = []
for i in range(1,100,2):
    train_acc, test_acc = model_fit_k_optimizer(i,X_scaled,y)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)
    
    


# In[69]:


train_acc_norm_list = []
test_acc_norm_list = []
for i in range(1,100,2):
    train_acc, test_acc = model_fit_k_optimizer(i,X_normalised,y)
    train_acc_norm_list.append(train_acc)
    test_acc_norm_list.append(test_acc)
    


# In[56]:


result = pd.DataFrame([np.arange(1,100,2),train_acc_list,test_acc_list]).T


# In[57]:


result.columns = ["k" , "Train_Acc", "Test_Acc"]


# In[58]:


result.head()


# In[59]:


result.plot(x='k', y=['Train_Acc', 'Test_Acc'], figsize=(10,5), grid=True)


# In[70]:


result = pd.DataFrame([np.arange(1,100,2),train_acc_norm_list,test_acc_norm_list]).T

result.columns = ["k" , "Train_Acc", "Test_Acc"]

result.head()

result.plot(x='k', y=['Train_Acc', 'Test_Acc'], figsize=(10,5), grid=True)


# In[80]:


def create_poly(train,test,degree):
    poly = PolynomialFeatures(degree=degree)
    train_poly = poly.fit_transform(train)
    test_poly = poly.fit_transform(test)
    return train_poly,test_poly


# In[100]:


train_X_poly, test_X_poly = create_poly(train_X, test_X, 3)


# In[101]:


knn.fit(train_X_poly,train_y)


# In[102]:


accuracy_score(train_y, knn.predict(train_X_poly))


# In[103]:


accuracy_score(test_y, knn.predict(test_X_poly))


# # Naive Bayes

# In[73]:


from sklearn.naive_bayes import GaussianNB


# In[74]:


nb = GaussianNB()


# In[78]:


model_fit(nb, X, y)


# In[79]:


model_fit(nb, X_normalised, y)


# In[ ]:




