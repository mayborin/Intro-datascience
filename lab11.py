
# coding: utf-8

# In[62]:

import pandas as pd
import numpy as np
import scipy


# In[3]:

df=pd.read_csv("/Users/LiuLuoming/Desktop/IDS/lab11_data/events_train.tsv",sep='\t',na_values=['-'])
pred=pd.read_csv("/Users/LiuLuoming/Desktop/IDS/lab11_data/prediction_trials.tsv",sep='\t',na_values=['-'])


# In[4]:

df=df[['closed_tstamp','event_type','latitude','longitude']]


# In[23]:

df[:0]


# In[5]:

df['closed_tstamp']=df['closed_tstamp'].str.extract('(....-..-..)', expand=True)
df['closed_tstamp']=df['closed_tstamp'].str.extract('(....)', expand=True)


# In[18]:

region=pred[:1][['nw_lon','nw_lat','se_lon','se_lat']].values[0]


# In[24]:

event=df[(df['longitude']>=region[0])&(df['longitude']<=region[2])&(df['latitude']>=region[3])&(df['latitude']<=region[1])]


# In[25]:

event


# In[47]:

group=event.groupby(['event_type'])


# In[52]:

arr=group.get_group('accidentsAndIncidents').groupby('closed_tstamp').size()


# In[57]:

arr


# In[58]:

arr[:len(arr)-1]


# In[66]:

from sklearn.linear_model import LinearRegression
x=arr[:len(arr)-1]
y=arr[1:]
lm=LinearRegression()
lm.fit(x,y)

