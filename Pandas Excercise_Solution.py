#!/usr/bin/env python
# coding: utf-8

# Import the pandas library  
# Read the data company.csv in pandas

# In[1]:


import pandas as pd
df = pd.read_csv("company.csv")


# Find the dimension of given data

# In[2]:


df.shape


# List down all the columns in data frame

# In[3]:


df.columns


# List the top 10 rows in datframe

# In[4]:


df.head()


# List the last 15 rows in datframe

# In[5]:


df.tail(15)


# Find the number of rows in dataframe

# In[6]:


df.shape[0]


# Check the information of dataframe

# In[7]:


df.info()


# Check the basic Statistics of Dataframe. Give your inference from the stats

# In[8]:


df.describe().T


# Retrieve 20 to 100 rows and Company and age usig iloc

# In[9]:


df.iloc[20:101,:2]


# Retrieve 20 to 100 rows and Company and age usig loc

# In[10]:


df.loc[20:100,["Company","Age"]]


# Chage the data in 100 row and Place as 'Noida'

# In[11]:


df.loc[100,"Place"]


# In[12]:


df.loc[100,"Place"] = "Noida"


# In[13]:


df.loc[100,:]


# Change the column 'Place' to 'City'

# In[14]:


df.rename(columns={'Place': 'City'}, inplace=True)


# In[15]:


df.columns


# List down the unique data in each columns and find length of unique data

# In[16]:


df.nunique()


# In[17]:


df.Company.unique()


# In[18]:


df.Company.value_counts()


# In[19]:


df.apply(lambda x:print(x.value_counts()))


# Rename all the possible labels of column in Company as three labels
# TCS
# CTS
# Infosys

# In[20]:


df.Company.replace({"Tata Consultancy Services":"TCS",
                    "Congnizant":"CTS","Infosys Pvt Lmt":"Infosys"},inplace=True)


# In[21]:


df.Company.value_counts()


# Where ever you see age as 0 replace with NA
# 
# ##Hint df[df.Age==20] = np.nan

# In[22]:


import numpy as np


# In[23]:


import numpy as np
df[df.Age==0] = np.nan


# In[24]:


df.Age.isna().sum()


# Check how many duplicated data is there?

# In[25]:


df.duplicated().sum()


# Remove all duplicated rowise data

# In[26]:


df = df.drop_duplicates()


# In[27]:


df.duplicated().sum()


# Remove the column 'Country'

# In[29]:


#df1 = df.drop(columns="Country")
del df["Country"]
df


# Remove the row number 137

# In[30]:


df.drop(index=136)


# Find number of each labels in Company

# In[31]:


df.Company.value_counts()


# Find number of each labels in City

# In[32]:


df.City.value_counts()


# Find Number of Null Values in each column

# In[33]:


df.isna().sum()


# Remove all Null values in Salary

# In[34]:


df.dropna(subset=["Salary"],inplace=True)


# Replace the Null values in Comapany with mode

# In[35]:


df.Company.fillna(df.Company.mode(),inplace=True)


# Replace the null value in Salary with median

# In[36]:


#df.Salary.fillna(df.Salary.median(),inplace=True)


# In[37]:


#df.Age.astype("float")
#df.loc[df['Age'].notnull(), 'Age'].apply(int)
#pd.to_numeric(df["Age"])
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df["Salary"] = pd.to_numeric(df["Salary"], errors='coerce')


# In[38]:


df.Age.fillna(df.Age.mean(),inplace=True)
df.Salary.fillna(df.Salary.median(),inplace=True)


# Replace the null value in age with mean

# Filter the data with age>40 and Salary<5000

# In[39]:


df[(df.Age>40) & (df.Salary<5000)]


# Draw an histogram chart for age, Salary

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.Age.plot(kind="hist")


# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.Salary.plot(kind="hist")


# Draw the pie chart for place and Company

# In[42]:


df.City.value_counts().plot(kind="pie")


# In[43]:


df.Company.value_counts().plot(kind="pie")


# In[44]:


df.columns


# Find the mean, count, median of salary with repective to the place

# In[47]:


df.groupby("City").Salary.agg(["mean","median","count"])


# In[48]:


df.groupby("City").Salary.agg(["mean","count","median"])


# Find the mean age with repective to the Company

# In[49]:


df.groupby("Company").Age.mean()


# In[50]:


df.dropna(inplace=True)


# In[51]:


df.head()


# In[52]:


df.head()


# In[53]:


df.isna().sum()


# In[54]:


num_col = df.select_dtypes(include=np.number).columns


# In[55]:


cat_col = df.select_dtypes(exclude=np.number).columns


# In[56]:


cat_col


# In[57]:


df[cat_col]


# In[58]:


#One hot encoding
encoded_cat_col = pd.get_dummies(df[cat_col])


# In[59]:


encoded_cat_col


# In[60]:


df_ready_model = pd.concat([df[num_col],encoded_cat_col], axis = 1)


# In[61]:


from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
for i in cat_col:
    df[i] = label_encoder.fit_transform(df[i])


# In[63]:


df.City.value_counts()


# In[64]:


#Performng standard scaling
from sklearn.preprocessing import StandardScaler
 
std_scale = StandardScaler().fit(df)
df_std = std_scale.transform(df)


# In[65]:


df_std


# In[66]:


#Performng standard scaling
from sklearn.preprocessing import MinMaxScaler

minmax_scale = MinMaxScaler().fit_transform(df)


# In[69]:


type(minmax_scale)


# In[70]:


pd.DataFrame(minmax_scale)


# In[71]:


col = ["Company","City"]
onehot = pd.get_dummies(df[col])


# In[72]:


col=["Age","Salary","Gender"]
dataf = pd.concat([df[col],onehot],axis=1)


# In[73]:


dataf


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[107]:


y_test = y_test.astype('int')


# In[100]:


from sklearn.linear_model import LogisticRegression


# In[ ]:





# In[ ]:





# In[102]:


model = LogisticRegression()

model.fit(x_train[["Age"]],y_train)


# In[113]:


a = model.predict(x_train[["Age"]])


# In[114]:


from sklearn.metrics import confusion_matrix


# In[115]:


confusion_matrix(a,y_train)


# In[ ]:




