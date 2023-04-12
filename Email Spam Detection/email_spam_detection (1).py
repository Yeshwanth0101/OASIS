#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import chardet   


# In[2]:


file_path = 'spam.csv'
with open (file_path, 'rb') as rawdata:
    result = chardet.detect(rawdata.read(500000))
print(result)     


# In[3]:


sms = pd.read_csv(file_path, encoding = "Windows-1252")


# In[4]:


sms.head()    


# In[5]:


print(sms.v2[71])
print(sms.v1[71])


# In[6]:


sms.info()    


# In[7]:


sms.dropna(how='any', inplace=True, axis=1)


# In[8]:


sms.columns=['label', 'message']
sms.head()


# In[9]:


sms.info()


# In[10]:


sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
print("After Modification:\n") 
sms.head()     


# In[11]:


print("Before Modification:\n") 
sms.head()    


# In[12]:


sms['message_len'] = sms.message.apply(len)
print("After Modification:\n") 
sms.head() 


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns  


# In[14]:


sns.set_style('whitegrid')         
plt.style.use('fivethirtyeight')   
plt.figure(figsize=(12,8))    


# In[15]:


sms[sms.label=='ham'].message_len.plot(bins=35, kind='hist', color='blue', label='Ham Messages', alpha=0.5)
sms[sms.label=='spam'].message_len.plot(kind='hist', color='red', label='Spam Messages', alpha=0.5)

plt.legend()
plt.xlabel("Message Length")     


# In[16]:


sms.describe()    


# In[17]:


sms[sms.label=='ham'].describe()


# In[18]:


import string
from nltk.corpus import stopwords


def temp_process(msg):
    
    STOPWORDS = stopwords.words('english')
    nopunc = [char for char in msg if char not in string.punctuation]  
    nopunc = ''.join(nopunc)
    nopunc = ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])
    return nopunc
     


# In[19]:


message = 'This is an example of a message. There are a lot of things here. Words, ALL the Words (AND information). Tons of Information!'


# In[20]:


print("MESSAGE:\n", message, "\n\n")    


# In[21]:


nopunc = [char for char in message if char not in string.punctuation]                   
print("Remove Punctuation:\n", nopunc, "\n\n")     


# In[22]:


nopunc = ''.join(nopunc)                                                             
print("After Join:\n", nopunc, "\n\n")


# In[23]:


print("Before Modification:\n") 
sms.head()     


# In[24]:


import nltk
nltk.download('stopwords')
  
sms['clean_msg'] = sms.message.apply(temp_process)
print("After Modification:\n") 
sms.head()


# In[25]:


h_words = sms[sms.label=='ham'].clean_msg.apply(lambda x: [word.lower() for word in x.split()])    
h_words[0]      


# In[26]:


from collections import Counter
ham_words = Counter()

for each_word in h_words:               
    ham_words.update(each_word)       
    
print(ham_words.most_common(50)) 


# In[27]:


s_words = sms[sms.label=='spam'].clean_msg.apply(lambda x: [word.lower() for word in x.split()]) 
spam_words = Counter()

for each_word in s_words:              
    spam_words.update(each_word)       
    
print(spam_words.most_common(50))    


# In[28]:


X = sms.clean_msg    
y = sms.label_num    

print(X.shape)       
print(y.shape) 


# In[29]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 42)     


# In[30]:


print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)


# In[31]:


from sklearn.feature_extraction.text import CountVectorizer    

vect = CountVectorizer()              
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)   


# In[32]:


X_test_dtm = vect.transform(X_test)


# In[33]:


X_test_dtm     


# In[34]:


X_train_dtm


# Naive Bayes Algorithm

# In[35]:


from sklearn.naive_bayes import MultinomialNB 
nb = MultinomialNB()         
nb.fit(X_train_dtm, y_train)


# # In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook.
# # On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.

# In[36]:


y_pred_class = nb.predict(X_test_dtm)     
y_pred_class     


# In[37]:


X_test[2:3]


# In[38]:


from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)


# In[40]:


metrics.confusion_matrix(y_test, y_pred_class)


# In[41]:


X_test[y_pred_class > y_test] 


# In[ ]:




