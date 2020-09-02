#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# 
# There are many models for **clustering** out there. In this notebook, we will be presenting the model that is considered the one of the simplest model among them. Despite its simplicity, the **K-means** is vastly used for clustering in many data science applications, especially useful if you need to quickly discover insights from **unlabeled data**. In this notebook, you learn how to use k-Means for customer segmentation.
# 
# Some real-world applications of k-means:
# - Customer segmentation
# - Understand what the visitors of a website are trying to accomplish
# - Pattern recognition
# - Machine learning
# - Data compression
# 
# 
# In this notebook we practice k-means clustering with 2 examples:
# - k-means on a random generated dataset
# - Using k-means for customer segmentation

# ### Import libraries
# Lets first import the required libraries.
# Also run <b> %matplotlib inline </b> since we will be plotting in this section.

# In[5]:


import random 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# # Customer Segmentation with K-Means
# Imagine that you have a customer dataset, and you need to apply customer segmentation on this historical data.
# Customer segmentation is the practice of partitioning a customer base into groups of individuals that have similar characteristics. It is a significant strategy as a business can target these specific groups of customers and effectively allocate marketing resources. For example, one group might contain customers who are high-profit and low-risk, that is, more likely to purchase products, or subscribe for a service. A business task is to retaining those customers. Another group might include customers from non-profit organizations. And so on.
# 
# Lets download the dataset. To download the data, we will use **`!wget`**. To download the data, we will use `!wget` to download it from IBM Object Storage.  
# __Did you know?__ When it comes to Machine Learning, you will likely be working with large datasets. As a business, where can you host your data? IBM is offering a unique opportunity for businesses, with 10 Tb of IBM Cloud Object Storage: [Sign up now for free](http://cocl.us/ML0101EN-IBM-Offer-CC)

# ### Load Data From CSV File  
# Before you can work with the data, you must use the URL to get the Cust_Segmentation.csv.

# In[1]:


import pandas as pd
cust_df = pd.read_csv("Cust_Segmentation.csv")
cust_df.head()


# In[2]:


cust_df.shape


# In[3]:


cust_df.duplicated().sum()


# In[4]:


cust_df.isna().sum()


# ### Pre-processing

# As you can see, __Address__ in this dataset is a categorical variable. k-means algorithm isn't directly applicable to categorical variables because Euclidean distance function isn't really meaningful for discrete variables. So, lets drop this feature and run clustering.

# In[6]:


df = cust_df.drop('Address', axis=1)
df.head()


# #### Normalizing over the standard deviation
# Now let's normalize the dataset. But why do we need normalization in the first place? Normalization is a statistical method that helps mathematical-based algorithms to interpret features with different magnitudes and distributions equally. We use __standardScaler()__ to normalize our dataset.

# In[7]:


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# ### Modeling

# In our example (if we didn't have access to the k-means algorithm), it would be the same as guessing that each customer group would have certain age, income, education, etc, with multiple tests and experiments. However, using the K-means clustering we can do all this process much easier.
# 
# Lets apply k-means on our dataset, and take look at cluster labels.

# In[8]:


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[9]:


df["clust_label"] = labels


# In[11]:


df.head(20)


# In[9]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# In[13]:


kn = KMeans(n_clusters=2)
kn.fit(X)


# In[15]:


kn.inertia_


# In[8]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Sum of within sum square')
pl.title('Elbow Curve')
pl.show()


# ### Insights
# We assign the labels to each row in dataframe.

# In[16]:


df["Clus_km"] = labels
df.head(5)


# We can easily check the centroid values by averaging the features in each cluster.

# In[50]:


df.groupby('Clus_km').mean()


# Now, lets look at the distribution of customers based on their age and income:

# In[51]:


area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)

plt.show()


# In[59]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
# plt.ylabel('Age', fontsize=18)
# plt.xlabel('Income', fontsize=16)
# plt.zlabel('Education', fontsize=16)
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

ax.scatter(X[:, 1], X[:, 0], X[:, 3], c= labels.astype(np.float))


# k-means will partition your customers into mutually exclusive groups, for example, into 3 clusters. The customers in each cluster are similar to each other demographically.
# Now we can create a profile for each group, considering the common characteristics of each cluster. 
# For example, the 3 clusters can be:
# 
# - AFFLUENT, EDUCATED AND OLD AGED
# - YOUNG AND LOW INCOME
# - MIDDLE AGED AND MIDDLE INCOME
# 

# In[ ]:




