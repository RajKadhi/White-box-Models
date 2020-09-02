#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


# In[47]:


arr1 = np.array([1,2,3.43543646456457575756])


# In[48]:


arr1


# In[45]:


arr1.dtype


# In[8]:


type(arr1)


# In[6]:


lst = [1,2,3]


# In[7]:


type(lst)


# In[9]:


arr1.shape


# In[10]:


arr1 = np.array([[1,2,3]])


# In[11]:


arr1.shape


# In[21]:


arr2 = np.array([[1,2,3],[4,5,6],[7,8,9],[11,12,14]])


# In[22]:


arr2


# In[14]:


arr2.shape


# In[17]:


arr2[0,2] = 999


# In[18]:


arr2


# In[23]:


arr2[:,:]


# In[20]:


arr2[:2,1:]


# In[24]:


arr2[:1,2:]


# In[25]:


arr3 = np.array([1,2,3,4,5,6,7,8,9,10,11,12])


# In[27]:


arr3.shape


# In[29]:


arr4 = arr3.reshape(3,4)


# In[35]:


arr4.reshape(12)


# In[36]:


np.linspace(0,1,10)


# In[38]:


np.empty([3,3])


# In[39]:


np.zeros([3,3])


# In[41]:


np.eye(3)


# In[71]:


a = np.random.rand(3,3)


# In[73]:


a


# In[72]:


np.array(a,dtype = np.int)


# In[76]:


a = np.array([[1,2,3],[4,5,6],[7,8,9]])


# In[77]:


b = np.array([[10,12,13],[14,15,16],[17,18,19]])


# In[78]:


a+b


# In[79]:


np.add(a,b)


# In[80]:


np.subtract(a,b)


# In[81]:


np.multiply(a,b)


# In[87]:


c = np.array([[1,2],[3,4],[7,8]])


# In[82]:


np.dot(a,b)


# In[84]:


a.shape


# In[88]:


c.shape


# In[90]:


d = np.dot(a,c)


# In[91]:


d


# In[95]:


#Broad Casting Feature
d * 10


# In[96]:


d


# In[102]:


e = np.array([[2,3]])


# In[109]:


d


# In[132]:


np.array([[28, 34],[61,76],[94,18]])


# In[138]:


np.append([[28, 34],[61,76],[94,18]],[[2,4]],axis = 0)


# In[135]:


np.append([[28, 34],[61,76],[94,18]],[[2],[4],[2]],axis = 1)


# In[121]:


a


# In[122]:


a.sum()


# In[124]:


a.sum(axis = 0)


# In[125]:


a.mean(axis = 0)


# In[128]:


a.max()


# In[92]:


l = [1,2,3,4,5]


# In[93]:


l + 10


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




