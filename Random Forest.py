#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from datetime import date
import seaborn as sns

import scipy.spatial as sp, scipy.cluster.hierarchy as hc
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, normalize 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import silhouette_score 
import scipy.cluster.hierarchy as shc

import random
from tqdm import tqdm

from scipy.cluster import hierarchy

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('matplotlib', 'inline')
sc = StandardScaler()
import warnings
warnings.filterwarnings("ignore")


# In[3]:


from sklearn import datasets


# In[4]:


#use dataset provided by Sci-kit learn module
print(datasets.load_boston().DESCR)


# In[5]:


Boston = pd.DataFrame(datasets.load_boston().data)
Boston.columns = datasets.load_boston().feature_names
Boston.head()


# In[6]:


#get the target variable
Boston_target = datasets.load_boston().target
print(Boston_target)


# In[ ]:


#Boston['MEDV'] = Boston_target


# In[7]:


x = Boston # Features
y = Boston_target  # Target


# In[8]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)


# In[21]:


# hyper-parameter space
param_grid_RF = {
    'n_estimators' : [10,20,50,100,200,500,1000],
    'max_features' : [0.6,0.8,"auto","sqrt"],
    'max_depth' : [2,4,5,6]
}

# build random forest model
rf_model = RandomForestRegressor(random_state=42,n_jobs=-1)
# gridsearch for the best hyper-parameter
gs_rf = GridSearchCV(rf_model, param_grid=param_grid_RF, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
# fit dataset
gs_rf.fit(x_train, y_train)

regressor = RandomForestRegressor(random_state=42, n_estimators=gs_rf.best_params_['n_estimators'], max_depth=gs_rf.best_params_['max_depth'], 
                                  max_features= gs_rf.best_params_['max_features']) 
regressor.fit(x_train, y_train)
    
y_pred = regressor.predict(x_test)
    
prediction_dict = pd.DataFrame(columns = ['real','predicted'])
prediction_dict.real = y_test
prediction_dict.predicted = y_pred

# RMSE (Root Mean Square Error)
from sklearn.metrics import mean_squared_error
rmse = float(format(np.sqrt(mean_squared_error(y_test, y_pred)),'.3f'))
print("\nRMSE:\n",rmse)
       
#feature importance
feature_importances = pd.DataFrame(regressor.feature_importances_,
                                         index=x_train.columns,columns=['importance']).sort_values('importance',ascending = False)   


# In[12]:


prediction_dict


# In[14]:


prediction_dict.plot()


# In[13]:


df = feature_importances.iloc[::-1]
df.plot.barh(stacked=True, figsize=(8,12));


# In[ ]:




