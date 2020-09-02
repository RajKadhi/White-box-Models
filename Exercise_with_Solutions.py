#!/usr/bin/env python
# coding: utf-8

# # Getting and Knowing your Data

# This time we are going to pull data directly from the internet.
# Special thanks to: https://github.com/justmarkham for sharing the dataset and materials.
# 
# ### Step 1. Import the necessary libraries

# In[1]:


import pandas as pd
import numpy as np


# ### Step 2. Import the dataset named "chipotle.tsv"

# ### Step 3. Assign it to a variable called chipo.

# In[52]:


chipo = pd.read_csv('chipotle.tsv', sep = '\t')


# In[53]:


chipo


# ### Step 4. See the first 10 entries

# In[3]:


chipo.head(10)
# chipo['choice_description'][4]


# ### Step 5. What is the number of observations in the dataset?

# In[4]:


chipo.info()#

# OR

chipo.shape[0]
# 4622 observations


# ### Step 6. What is the number of columns in the dataset?

# In[5]:


chipo.shape[1]


# ### Step 7. Print the name of all the columns.

# In[6]:


chipo.columns


# ### Step 8. How is the dataset indexed?

# In[7]:


chipo.index


# ### Step 9. Which was the most ordered item? 

# In[7]:


data = chipo.groupby("item_name")["quantity"].sum()


# In[15]:


data = data.sort_values(ascending = False)


# In[18]:


pd.DataFrame(data).head(1)


# In[29]:


or

#chipo.groupby('item_name')['quantity'].sum().sort_values(ascending=False).head(1)


# ### Step 10. How many items were ordered?

# In[19]:


data = chipo.groupby("item_name")["quantity"].sum()


# In[20]:


data.sort_values(ascending = False)[0]


# ### Step 11. What was the most ordered item in the choice_description column?

# In[21]:


data = chipo.groupby("choice_description")["quantity"].sum()

data = data.sort_values(ascending = False)

pd.DataFrame(data).head(1)


# ### Step 12. How many items were orderd in total?

# In[42]:


total_items_orders = chipo.quantity.sum()
total_items_orders


# ### Step 13. Turn the item price into a float

# In[29]:


chipo["item_price"] = chipo["item_price"].apply(lambda x : x[1:]).astype('float')


# In[43]:


#dollarizer = lambda x: float(x[1:-1])
#chipo.item_price = chipo.item_price.apply(dollarizer)


# ### Step 14. How much was the revenue for the period in the dataset?

# In[54]:


revenue = (chipo["quantity"] * chipo["item_price"]).sum()


# In[55]:


print('Revenue was: $' + str(np.round(revenue,2)))


# ### Step 15. How many orders were made in the period?

# In[39]:


chipo["order_id"]


# In[56]:


#or
#chipo.order_id.value_counts().sum()


# ### Step 16. What is the average amount per order?

# In[57]:


chipo['revenue'] = chipo['quantity']* chipo['item_price']


# In[58]:


chipo


# In[59]:


chipo.groupby('order_id')['revenue'].sum().mean()


# ### Step 17. How many different items are sold?

# In[50]:


chipo.item_name.nunique()


# In[ ]:




