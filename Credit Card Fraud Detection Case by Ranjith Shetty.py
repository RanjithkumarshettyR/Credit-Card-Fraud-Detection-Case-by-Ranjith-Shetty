#!/usr/bin/env python
# coding: utf-8

# Credit Card Fraud Detection Case by Ranjith Shetty

# In[ ]:





# Imports

# In[1]:


import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sys
import matplotlib.pyplot as plt
import seaborn as sns


# Data Importing

# In[4]:


data = pd.read_csv(r"C:\Users\ADMIN\Documents\Data sets\Credit Card Fraud Detection/creditcard1.csv")


# In[ ]:





# Exploring the Dataset

# In[5]:


print(data.columns)


# In[6]:


data.shape


# In[7]:


# random_state helps assure that you always get the same output when you split the data
# this helps create reproducible results and it does not actually matter what the number is
# frac is percentage of the data that will be returned
data = data.sample(frac = 0.2, random_state = 1)
print(data.shape)


# In[ ]:





# In[8]:


# plot the histogram of each parameter
data.hist(figsize = (20, 20))
plt.show()


# You can see most of the V's are clustered around 0 with some or no outliers. Notice we have very few fraudulent cases over valid cases in our class histogram.

# In[ ]:





# In[ ]:





# In[9]:


# determine the number of fraud cases
fraud = data[data['Class'] == 1]
valid = data[data['Class'] == 0]

outlier_fraction = len(fraud) / float(len(valid))
print(outlier_fraction)

print('Fraud Cases: {}'.format(len(fraud)))
print('Valid Cases: {}'.format(len(valid)))


# In[ ]:





# In[10]:


# correlation matrix
corrmat = data.corr()
fig = plt.figure(figsize = (12, 9))

sns.heatmap(corrmat, vmax = .8, square = True)
plt.show()


# Organizing the Data

# In[11]:


# get the columns from the dataframe
columns = data.columns.tolist()

# filter the columns to remove the data we do not want
columns = [c for c in columns if c not in ['Class']]

# store the variable we will be predicting on which is class
target = 'Class'

# X includes everything except our class column
X = data[columns]
# Y includes all the class labels for each sample
# this is also one-dimensional
Y = data[target]

# print the shapes of X and Y
print(X.shape)
print(Y.shape)


# In[ ]:





# Fit The Model

# In[18]:


n_outliers = len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    
    # fit the data and tag outliers
    if clf_name == 'Local Outlier Factor':
        y_pred = clf.fit_predict(X)
        scores_pred = clf.negative_outlier_factor_
    else:
        clf.fit(X)
        scores_pred = clf.decision_function(X)
        y_pred = clf.predict(X)
        
    # reshape the prediction values to 0 for valid and 1 for fraud
    y_pred[y_pred == 1] = 0
    y_pred[y_pred == -1] = 1

    # calculate the number of errors
    n_errors = (y_pred != Y).sum()
    
    # classification matrix
    print('{}: {}'.format(clf_name, n_errors))
    print(accuracy_score(Y, y_pred))
    print(classification_report(Y, y_pred))


# In[ ]:





# Conclusion

# Looking at precision for fraudulent cases (1) lets us know the percentage of cases that are getting correctly labeled.Precision accounts for false-positives.Recall accounts for false-negatives. Low numbers could mean that we are constantly calling clients asking them if they actually made the transaction which could be annoying.
# 
# Goal: To get better percentages.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




