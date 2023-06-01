
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('insta.csv')
x= df.drop(['fake'],axis=1)
y= df.fake

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=42)

clf_entropy = DecisionTreeClassifier(criterion = "entropy") 
clf_entropy.fit(x_train,y_train) 
clf_entropy.predict(x_test) 

pickle.dump((clf_entropy), open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

