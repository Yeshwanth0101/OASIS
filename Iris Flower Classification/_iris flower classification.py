#!/usr/bin/env python
# coding: utf-8

# In[31]:


import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# In[32]:


df=pd.read_csv('iris.csv')


# In[33]:


df


# In[34]:


df.isnull().sum().sum()


# In[35]:


df.shape


# In[36]:


df.describe()


# In[37]:


df.dtypes


# In[38]:


df.head


# In[39]:


df.tail


# In[40]:


df.info


# In[41]:


df.describe


# In[42]:


df.nunique


# In[43]:


df.count


# In[44]:


df['Species'].unique()


# In[45]:


df['Species'].value_counts()


# In[46]:


import matplotlib.pyplot as mat
import seaborn as sb
mat.figure(figsize=(15,15))
sb.countplot('Species',data=df,palette='hls')


# In[47]:


sb.pairplot(df)


# In[48]:


df.corr()


# In[49]:


mat.figure(figsize=(15,12))
sb.heatmap(df.corr(),annot=True)
mat.show()


# In[50]:


x=df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y=df['Species']


# In[51]:


print(x)


# In[52]:


print(y)


# In[53]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)


# In[54]:


print(X_train)


# In[55]:


print(X_test)


# In[56]:


print(y_train)


# In[57]:


print(y_test)


# In[58]:


from sklearn.linear_model import LogisticRegression

model1 = LogisticRegression()

model1.fit(X_train, y_train)

model1.score(X_test, y_test)


# In[59]:


from sklearn.neighbors import KNeighborsClassifier

model_KNN = KNeighborsClassifier(n_neighbors=10)

model_KNN.fit(X_train, y_train)

model_KNN.score(X_test, y_test)


# In[60]:


model1.predict([[5.1, 3.8, 1.9, 0.4]])


# In[61]:


model_KNN.predict([[7,3.2,4.7,1.4]])


# In[65]:


model1.predict([[7,3.2,4.7,1.4]])


# In[63]:


model_KNN.predict([[6.4,2.8,5.6,2.1]])


# In[64]:


df.loc[50]


# In[ ]:




