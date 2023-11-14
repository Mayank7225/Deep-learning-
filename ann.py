#!/usr/bin/env python
# coding: utf-8

# In[31]:


get_ipython().system('pip install tensorflow')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn


# In[32]:


dataset=pd.read_csv('Churn_modelling.csv')
X=dataset.iloc[:,3:13]
y=dataset.iloc[:,13]


# In[33]:


#create dummy varibales
geography=pd.get_dummies(X["Geography"],drop_first=True)
gender=pd.get_dummies(X["Gender"],drop_first=True)


# In[34]:


#concatenate the data Frames
X=pd.concat([X,geography,gender],axis=1)


# In[35]:


#drop the unnecessary clomns for that purpose now 
X=X.drop(['Geography','Gender'],axis=1)


# In[36]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


# In[37]:


from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[38]:


import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU,PReLU,ELU
from keras.layers import Dropout


# In[41]:


#initializaton
classifier=Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu',input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'he_uniform',activation='relu'))
# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'glorot_uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'Adamax', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100)


# In[44]:


print(model_history.history.keys())
# summarize history for accuracy
plt.plot(model_history.history['accuracy'])
plt.plot(model_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# # as we can see that we have train accuracy greater than the test accuracy ,it means that our model might be having following problems
# 1)overfitting 
# 2)Model achitecture may be too complex for the given data sets
# 3)we might as well have given the insufficient refularization
# 4)we have limited data for the model to learn
# 5)data mismatch  we might have used different data for testign and different for training
# 
# # To address these issues we can resort to following ways 
# 1)implenting a L2 regularisaton
# 2)simplify model as in this case we can use the simple i hidden layer model for predicton 
# 3) data augmentation
# 4)increase the data
# 5) cross validation so that the corect data is fed to the test as well as the train model
# 

# In[45]:


# summarize history for loss
plt.plot(model_history.history['loss'])
plt.plot(model_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show() 


# In[46]:


# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Calculate the Accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_pred,y_test) 


# In[ ]:




