#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
df = pd.read_csv("/Users/raj/ML Inceptaz classes/Noor/Nov 23/insurance.csv")
df.shape
df.head()


# In[5]:


df.shape


# In[10]:


df.info()
df.describe()


# In[16]:


#dup_bool = df.duplicated()
df1 = df.drop_duplicates( subset=None, keep='first' )


# In[22]:


# Importing Libraies and data set
import pandas as pd
import numpy as np

#Reading the dataset in dataframe using pandas
insurance =pd.read_csv('https://raw.githubusercontent.com/nursnaaz/25DaysInMachineLearning/master/18%20-%20Day%20-%2018%20-%20Linear%20Regression%20Practise%20Python/Kaggle%20Assignment/insurance.csv')

#Script to find the outliers
for col_name in insurance.select_dtypes(include=np.number).columns[:-1]:
    print(col_name)
    q1 = insurance[col_name].quantile(0.25)
    q3 = insurance[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr
    print("Outliers = ",insurance.loc[(insurance[col_name] < low) | (insurance[col_name] > high), col_name])
    
 

#Script to exclude the outliers
for col_name in insurance.select_dtypes(include=np.number).columns[:-1]:
    print(col_name)
    q1 = insurance[col_name].quantile(0.25)
    q3 = insurance[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr 
    print("Exclude the Outliers = ",insurance.loc[~((insurance[col_name] < low) | (insurance[col_name] > high)), col_name])
    insurance[col_name] = insurance.loc[~((insurance[col_name] < low) | (insurance[col_name] > high)), col_name]
    
    
    
#Script to impute the outliers with median
for col_name in insurance.select_dtypes(include=np.number).columns[:-1]:
    print(col_name)
    q1 = insurance[col_name].quantile(0.25)
    q3 = insurance[col_name].quantile(0.75)
    iqr = q3 - q1
    
    low = q1-1.5*iqr
    high = q3+1.5*iqr 
    print("Change the outliers with median ",insurance[col_name].median())
    insurance.loc[(insurance[col_name] < low) | (insurance[col_name] > high), col_name] = insurance[col_name].median()
    
#Detecting outlier with z-score
outliers=[]
def detect_outlier(insurance):
    
    threshold=3
    mean_1 = np.mean(insurance)
    std_1 = np.std(insurance)
    
    
    for y in insurance:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers

outlier_datapoints = insurance.select_dtypes(include=np.number).apply(detect_outlier)


# In[23]:


insurance


# In[28]:


df = insurance
df.head()


# In[30]:


y = df["expenses"]
y.shape


# In[35]:


#xall = df[["age"],["sex"],["bmi"],["children"],["smoker"],["region"]]
xall = df[["age" , "sex"]]
xall.shape


# In[39]:


df["region"].value_counts()


# In[40]:


df["region"].isna().sum()


# In[ ]:




