#!/usr/bin/env python
# coding: utf-8

# In[91]:


import numpy as np
import pandas as pd 
import chardet   


# In[92]:


file_path = 'spam.csv'
with open (file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(500000))
print(result)     


# In[93]:


sms = pd.read_csv(file_path, encoding = "Windows-1252")


# In[94]:


sms.head()    


# In[95]:


print(sms.v2[71])
print(sms.v1[71])


# In[96]:


sms.info()    


# In[97]:


sms.dropna(how='any', inplace=True, axis=1)


# In[98]:


sms.columns=['label', 'message']
sms.head()


# In[99]:


sms.info()


# In[100]:


sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
print("After Modification:\n") 
sms.head()     


# In[101]:


print("Before Modification:\n") 
sms.head()    


# In[102]:


sms['message_len'] = sms.message.apply(len)
print("After Modification:\n") 
sms.head() 


# In[103]:


import matplotlib.pyplot as plt
import seaborn as sns  


# In[104]:


sns.set_style('whitegrid')         
plt.style.use('fivethirtyeight')   
plt.figure(figsize=(12,8))    


# In[105]:


sms[sms.label=='ham'].message_len.plot(bins=35, kind='hist', color='blue', label='Ham Messages', alpha=0.5)
sms[sms.label=='spam'].message_len.plot(kind='hist', color='red', label='Spam Messages', alpha=0.5)

plt.legend()
plt.xlabel("Message Length")     


# In[106]:


sms.describe()    


# In[107]:


sms[sms.label=='ham'].describe()


# In[108]:


import string
from nltk.corpus import stopwords


def temp_process(msg):
    
    STOPWORDS = stopwords.words('english')
    nopunc = [char for char in msg if char not in string.punctuation]  
    nopunc = ''.join(nopunc)
    nopunc = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    return nopunc
     


# In[109]:


message = 'This is an example of a message. There are a lot of things here. Words, ALL the Words (AND information). Tons of Information!'


# In[110]:


print("MESSAGE:\n", message, "\n\n")    


# In[111]:


nopunc = [char for char in message if char not in string.punctuation]                   
print("Remove Punctuation:\n", nopunc, "\n\n")     


# In[112]:


nopunc = ''.join(nopunc)                                                             
print("After Join:\n", nopunc, "\n\n")


# In[113]:


print("Before Modification:\n") 
sms.head()     


# In[ ]:




