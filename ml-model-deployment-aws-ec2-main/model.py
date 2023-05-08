#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
from sklearn.ensemble import RandomForestClassifier


# In[3]:


sonar = pd.read_csv('sonar.csv', header=None)


# In[4]:


sonar.head()


# In[5]:


sonar.shape


# In[6]:


sonar.describe()


# In[7]:


sonar[60].value_counts()


# In[8]:


sonar.groupby(60).mean()


# In[9]:


# separating data and Labels
X = sonar.drop(columns=60, axis=1)
Y = sonar[60]


# In[10]:


print(X)
print(Y)


# In[ ]:





# In[11]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)


# In[12]:


print(X.shape, X_train.shape, X_test.shape)


# In[13]:


print(X_train)
print(Y_train)


model = LogisticRegression()



#training the Logistic Regression model with training data
model.fit(X_train, Y_train)

#accuracy on training data
X_train_prediction = model.predict(X_train)
#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)*100



pickle.dump(model,open('model.pkl','wb'))
pickle.dump(test_data_accuracy,open('accuracy.pkl','wb'))
