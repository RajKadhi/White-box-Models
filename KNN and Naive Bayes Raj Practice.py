#!/usr/bin/env python
# coding: utf-8

# In[7]:


# import libraries 
import pandas as pd
import numpy as np


# In[20]:


df = pd.read_csv('/Users/raj/ML Inceptaz classes/Noor/Dec7/bankmarketing/bank-additional-full.csv',sep = ';')


# In[22]:


df.shape


# In[24]:


df.duplicated().sum()


# In[31]:


df.drop_duplicates(inplace = True)


# In[32]:


df.duplicated().sum()


# In[17]:


df.isna().sum()


# In[4]:


from imblearn.under_sampling import NearMiss


# In[3]:


pip install imblearn


# In[5]:


undersample = NearMiss()


# In[ ]:


x_undersample, y_undersample = undersample.fit_sample()


# In[6]:


x = df.drop['y']


# In[8]:


df = pd.read_csv("/Users/raj/ML Inceptaz classes/Noor/Dec7/K-means, Heirarchical/Hierrachical/MallCustomers.csv")


# In[9]:


df.head()


# In[11]:


df.drop(columns = ["CustomerID","Gender","Age"], inplace = True)


# In[12]:


df.he


# In[14]:


import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[15]:



plt.figure(figsize=(8,5))
plt.title("Annual income distribution",fontsize=16)
plt.xlabel ("Annual income (k$)",fontsize=14)
plt.grid(True)
plt.hist(df['Annual Income (k$)'],color='orange',edgecolor='k')
plt.show()


# In[16]:


plt.figure(figsize=(8,5))
plt.title("Spending Score distribution",fontsize=16)
plt.xlabel ("Spending Score (1-100)",fontsize=14)
plt.grid(True)
plt.hist(df['Spending Score (1-100)'],color='green',edgecolor='k')
plt.show()


# In[17]:


plt.figure(figsize=(8,5))
plt.title("Annual Income and Spending Score correlation",fontsize=18)
plt.xlabel ("Annual Income (k$)",fontsize=14)
plt.ylabel ("Spending Score (1-100)",fontsize=14)
plt.grid(True)
plt.scatter(df['Annual Income (k$)'],df['Spending Score (1-100)'],color='red',edgecolor='k',alpha=0.6, s=100)
plt.show()


# In[21]:


import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
#plt.grid(True)
dendrogram = sch.dendrogram(sch.linkage(df, method = 'centroid'))
plt.show()


# In[27]:


plt.figure(figsize=(15,6))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.hlines(y=30,xmin=0,xmax=2000,lw=3,linestyles='--')
plt.text(x=900,y=20,s='Horizontal line crossing 7 vertical lines',fontsize=20)
#plt.grid(True)
dendrogram = sch.dendrogram(sch.linkage(df, method = 'centroid'))
plt.show()


# In[ ]:




