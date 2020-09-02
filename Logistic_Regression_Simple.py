#!/usr/bin/env python
# coding: utf-8

# Import the pandas library  
# Read the data company.csv in pandas

# In[1]:


import pandas as pd
df = pd.read_csv("company.csv")


# In[2]:


df


# Find the dimension of given data

# In[3]:


df.shape


# List down all the columns in data frame

# In[4]:


df.columns


# List the top 10 rows in datframe

# In[5]:


df.head()


# List the last 15 rows in datframe

# In[6]:


df.tail(15)


# Find the number of rows in dataframe

# In[7]:


df.shape[0]


# Check the information of dataframe

# In[8]:


df.info()


# Check the basic Statistics of Dataframe. Give your inference from the stats

# In[9]:


df.describe().T


# Retrieve 20 to 100 rows and Company and age usig iloc

# In[10]:


df.iloc[20:101,:2]


# Retrieve 20 to 100 rows and Company and age usig loc

# In[11]:


df.loc[20:100,["Company","Age"]]


# Chage the data in 100 row and Place as 'Noida'

# In[12]:


df.loc[100,"Place"]


# In[13]:


df.loc[100,"Place"] = "Noida"


# In[14]:


df.loc[100,:]


# Change the column 'Place' to 'City'

# In[15]:


df.rename(columns={'Place': 'City'}, inplace=True)


# In[16]:


df.columns


# List down the unique data in each columns and find length of unique data

# In[17]:


df.nunique()


# In[18]:


df.Company.unique()


# In[19]:


df.Company.value_counts()


# In[20]:


df.apply(lambda x:print(x.value_counts()))


# Rename all the possible labels of column in Company as three labels
# TCS
# CTS
# Infosys

# In[21]:


df.Company.replace({"Tata Consultancy Services":"TCS",
                    "Congnizant":"CTS","Infosys Pvt Lmt":"Infosys"},inplace=True)


# In[22]:


df.Company.value_counts()


# Where ever you see age as 0 replace with NA
# 
# ##Hint df[df.Age==20] = np.nan

# In[23]:


import numpy as np


# In[24]:


import numpy as np
df[df.Age==0] = np.nan


# In[25]:


df.Age.isna().sum()


# Check how many duplicated data is there?

# In[26]:


df.duplicated().sum()


# Remove all duplicated rowise data

# In[27]:


df = df.drop_duplicates()


# In[28]:


df.duplicated().sum()


# In[29]:


df.shape


# Remove the column 'Country'

# In[30]:


#df1 = df.drop(columns="Country")
del df["Country"]
df


# Remove the row number 137

# In[31]:


df.drop(index=136)


# Find number of each labels in Company

# In[32]:


df.Company.value_counts()


# Find number of each labels in City

# In[33]:


df.City.value_counts()


# Find Number of Null Values in each column

# In[34]:


df.isna().sum()


# Remove all Null values in Salary

# In[35]:


df.dropna(subset=["Salary"],inplace=True)


# Replace the Null values in Comapany with mode

# In[36]:


df.Company.fillna(df.Company.mode(),inplace=True)


# Replace the null value in Salary with median

# In[37]:


#df.Salary.fillna(df.Salary.median(),inplace=True)


# In[38]:


#df.Age.astype("float")
#df.loc[df['Age'].notnull(), 'Age'].apply(int)
#pd.to_numeric(df["Age"])
df["Age"] = pd.to_numeric(df["Age"], errors='coerce')
df["Salary"] = pd.to_numeric(df["Salary"], errors='coerce')


# In[39]:


df.Age.fillna(df.Age.mean(),inplace=True)
df.Salary.fillna(df.Salary.median(),inplace=True)


# Replace the null value in age with mean

# Filter the data with age>40 and Salary<5000

# In[40]:


df[(df.Age>40) & (df.Salary<5000)]


# Draw an histogram chart for age, Salary

# In[41]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.Age.plot(kind="hist")


# In[42]:


get_ipython().run_line_magic('matplotlib', 'inline')
df.Salary.plot(kind="hist")


# Draw the pie chart for place and Company

# In[43]:


df.City.value_counts().plot(kind="pie")


# In[44]:


df.Company.value_counts().plot(kind="pie")


# In[45]:


df.columns


# Find the mean, count, median of salary with repective to the place

# In[46]:


df.groupby("City").Salary.agg(["mean","median","count"])


# In[47]:


df.groupby("City").Salary.agg(["mean","count","median"])


# Find the mean age with repective to the Company

# In[48]:


df.groupby("Company").Age.mean()


# In[49]:


df.dropna(inplace=True)


# In[50]:


df.head()


# In[51]:


df.head()


# In[52]:


df.isna().sum()


# In[53]:


df.isna().sum()


# In[54]:


df.head()


# In[55]:


for i in df.columns:
    print(df[i].max())


# In[56]:


df.apply(lambda x : x.max(),axis=0)


# In[57]:


df.corr()


# In[58]:


df["Age"] = df["Age"].astype('int')
df["Salary"] = df["Salary"].astype('int')
df["Gender"] = df["Gender"].astype('int')


# In[61]:


df.head()


# In[62]:


df.shape


# In[64]:


cat_column_df = df.select_dtypes(exclude = np.number)


# In[66]:


cat_column_df_encoded = pd.get_dummies(cat_column_df)


# In[67]:


num_column_df = df.select_dtypes(include = np.number)


# In[68]:


num_column_df


# In[70]:


df_processed = pd.concat([num_column_df,cat_column_df_encoded],axis="columns")


# In[72]:


X = df_processed.drop(columns="Gender")


# In[74]:


y = df_processed["Gender"]


# In[75]:


from sklearn.model_selection import train_test_split


# In[76]:


train_X, test_X, train_y, test_y = train_test_split(X,y, test_size=0.3, random_state = 8)


# In[77]:


train_X.shape


# In[78]:


test_X.shape


# In[79]:


train_y.shape


# In[80]:


test_y.shape


# In[81]:


from sklearn.linear_model import LogisticRegression


# In[82]:


log_model = LogisticRegression()


# In[83]:


log_model.fit(train_X, train_y)


# In[84]:


log_model.intercept_


# In[85]:


log_model.coef_


# In[88]:


train_pred = log_model.predict(train_X)


# In[1]:


log_model.predict_proba(train_X)


# In[95]:


from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, classification_report


# In[90]:


confusion_matrix(train_y,train_pred )


# In[92]:


accuracy_score(train_y,train_pred)


# In[94]:


recall_score(train_y,train_pred)


# In[97]:


print(classification_report(train_y,train_pred))


# In[98]:


test_pred = log_model.predict(test_X)


# In[100]:


print("Confusion Matrix :",confusion_matrix(test_y,test_pred ))

print(" Test Accuracy :",accuracy_score(test_y,test_pred))

print(" Test Recall :",recall_score(test_y,test_pred))

print(classification_report(test_y,test_pred))


# In[105]:


import matplotlib.pyplot as plt 
plt.rc("font", size=14)
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_y, log_model.predict(test_X))
fpr, tpr, thresholds = roc_curve(test_y, log_model.predict_proba(test_X)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[60]:


X = df[["Age"]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[204]:


X.shape


# In[169]:


y = df["Salary"]


# In[167]:


from sklearn.model_selection import train_test_split


# In[171]:


X.shape


# In[191]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 8 )


# In[196]:


train_X.shape,test_X.shape, train_y.shape, test_y.shape


# In[205]:


from sklearn.linear_model import LinearRegression
lin = LinearRegression()


# In[206]:


lin.fit(train_X,train_y)


# In[207]:


lin.intercept_


# In[208]:


lin.coef_


# In[211]:


train_pred = lin.predict(train_X)


# In[212]:


from sklearn.metrics import mean_squared_error


# In[215]:


print("Train MSE :",mean_squared_error(train_y,train_pred))


# In[217]:


test_pred = lin.predict(test_X)


# In[219]:


print("Test MSE :",mean_squared_error(test_y,test_pred))


# In[220]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[225]:


plt.scatter(x = train_X,y = train_y,c = "red")
plt.plot(train_X,train_pred,'b')


# In[ ]:





# In[226]:


df.columns


# In[229]:


cat_col = df.select_dtypes(exclude= np.number).columns


# In[230]:


num_col = df.select_dtypes(include= np.number).columns


# In[232]:


cat_col_num = pd.get_dummies(df[cat_col])


# In[239]:


cat_col_num 


# In[240]:


df[num_col]


# In[236]:


df_new = pd.concat([df[num_col],cat_col_num], axis = "columns")


# In[237]:


df_new.shape


# In[243]:


X = df_new.drop(columns = "Salary")


# In[244]:


y = df_new["Salary"]


# In[245]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size = 0.3, random_state = 8 )


# In[246]:


lin = LinearRegression()

lin.fit(train_X,train_y)


# In[248]:


train_pred = lin.predict(train_X)


# In[249]:


mean_squared_error(train_y,train_pred)


# In[250]:


test_pred = lin.predict(test_X)

mean_squared_error(test_y,test_pred)


# In[252]:


mean_squared_error([2],[5])


# In[253]:


lin.coef_


# In[256]:


list(zip(train_X.columns,lin.coef_))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[85]:


import numpy as np


# In[89]:


cat_col = df.select_dtypes(exclude=np.number).columns


# In[91]:


df_cat_num = pd.get_dummies(df[cat_col])


# In[92]:


num_col = df.select_dtypes(include=np.number).columns


# In[93]:


num_col


# In[96]:


df_new = pd.concat([df[num_col], df_cat_num], axis = "columns")


# In[106]:


X = df_new[["Age"]]


# In[111]:


X.shape


# In[107]:


y = df_new["Salary"]


# In[112]:


y.shape


# In[108]:


from sklearn.model_selection import train_test_split


# In[127]:


train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,random_state = 90)


# In[128]:


train_X.shape


# In[129]:


test_X.shape


# In[130]:


X.shape


# In[132]:


train_y.shape


# In[133]:


test_y.shape


# In[134]:


y.shape


# In[135]:


from sklearn.linear_model import LinearRegression


# In[136]:


linear = LinearRegression()


# In[137]:


linear.fit(train_X,train_y)


# In[138]:


linear.intercept_


# In[139]:


linear.coef_


# In[ ]:


Salary of a Person whose age is 30

Salary = 6863 - 56.02(30)

5183.93


# In[140]:


linear.predict([[30]])


# In[141]:


from sklearn.metrics import mean_squared_error


# In[148]:


train_pred = linear.predict(train_X)


# In[153]:


train_pred


# In[144]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[147]:


plt.scatter(df["Age"],df["Salary"],color = "black")
plt.xlabel("Age")
plt.ylabel("Salary")


# In[155]:


plt.scatter(train_X["Age"],train_y,color = "black")

plt.plot(train_X["Age"],train_pred,'b')



plt.xlabel("Age")
plt.ylabel("Salary")


# In[156]:


#train MSE
mean_squared_error(train_y,train_pred)


# In[ ]:


#test MSE


# In[157]:


test_pred = linear.predict(test_X)


# In[158]:


#test MSE
mean_squared_error(test_y,test_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[165]:


X = df_new.drop(columns="Salary")

X.shape

y = df_new["Salary"]

y.shape

from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3,random_state = 30)

train_X.shape

test_X.shape

X.shape

train_y.shape

test_y.shape

y.shape

from sklearn.linear_model import LinearRegression

linear = LinearRegression()

linear.fit(train_X,train_y)

linear.intercept_

linear.coef_




from sklearn.metrics import mean_squared_error

train_pred = linear.predict(train_X)

train_pred





#train MSE
print("Train MSE: ",mean_squared_error(train_y,train_pred))

#test MSE

test_pred = linear.predict(test_X)

#test MSE
print("Test MSE :",mean_squared_error(test_y,test_pred))

