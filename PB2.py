#!/usr/bin/env python
# coding: utf-8

# In[19]:


#list
#List are mutable, that is value of elements of list can be altered
#The index starts from 0


a = [1,2,2.1,"Hello",1+2j]
print(a)

print(a[3])

print ("a[2] is : ",a[2])
a[2] = 5.1
print ("a[2] is : ",a[2])

print ("My list is: ",a)

a[1] = (2,3,4,5)

print ("My list is: ",a)

a.insert(5,2)
a.insert(0,2)

print ("My list is: ",a)


# In[17]:


#Tuples
#Tuples are immutable, that is value of elements of tuples cannot be altered


a = (5,"Hello", 23.2, 1+2j)

print "Tuple a: ",a

print "a[2] : ",a[2]

a[2] = 3.3

#a[4] = 23


# In[25]:


#String

#String is a immutable

s = "Hi, How are you?"
print s

s = "Hello, how are you?"
print s

print s[4]

s[4] = "s"


# In[5]:


#Set

#set is unordered collection of unique items

#Indexing has no meaning, Hence slicing operator[] has no work

a = {1,1,1,1,1,1,1,2,6,8,6,3,123,124}
print (a)

type(a)

a.pop()

a[0]


# In[25]:


s = {'q','a','e','r','b','b','a'}

print(s)
s.pop()
s.add('t')
print(s)


# In[12]:


#Dictionary

#Dictionary is an unordered collection of key-value pair

#Dictionary is mutable

dictionary = {1:'value', "key":2,2:3,'l':[1,2,3,4]}

print(type(dictionary))

print (dictionary['key'])

print (dictionary[2])

dictionary[2] = 4

print (dictionary)

print(dictionary['l'][2])


# In[ ]:




