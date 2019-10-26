# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:49:38 2019

@author: SEJAL
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import statsmodels.api as sm

data=pd.read_csv("C:/Users/SEJAL/Desktop/XL_Files/Air_quality.csv",encoding = "ISO-8859–1")
print (data.shape)
print (data.rspm.isna().sum())
print (data.spm.isna().sum())
print (data.so2.isna().sum())
print (data.no2.isna().sum())
print (data.date.isna().sum())
print (data.columns)
#%%

data=data.drop(['stn_code','sampling_date','state','x','pm2_5','location_monitoring_station'],axis=1)

print (data.head(10))
print (data.shape)

#%%
print (data.location.unique())
#%%
data.date.dropna()
data["type"].fillna("not defined", inplace = True)

data['location']=data['location'].replace('Aurangabad (MS)','Aurangabad') 

data['location']=data['location'].replace('Bombay','Mumbai') 

data['location']=data['location'].replace('Chandrapur','Chandarpur') 

data["type"]= data["type"].replace('not defined', "others")
data["type"]= data["type"].replace('Sensitive Area', "others")
data["type"]= data["type"].replace('Sensitive Areas', "others")

data["type"]= data["type"].replace('Industrial Area', "Industrial Areas")
data["type"]= data["type"].replace('Industrial', "Industrial Areas")

data["type"]= data["type"].replace('Residential and others', "Residential")
data["type"]= data["type"].replace('Residential, Rural and other Areas', "Residential")



data['date'] = pd.to_datetime(data['date'],format='%d-%m-%Y') # date parse
data['year'] = data['date'].dt.year # year


#%%

data[['so2','location']].groupby(["location"]).median().sort_values(by='so2',ascending=False).plot.bar(color='r')
plt.show()
#%%
df = data[['so2','year','location']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='so2', data=df)
#%%
data[['so2','type']].groupby(["type"]).median().sort_values(by='so2',ascending=False).plot.bar(color='r')
plt.show()
#%%
data[['no2','location']].groupby(["location"]).median().sort_values(by='no2',ascending=False).plot.bar(color='r')
plt.show()
#%%
df = data[['no2','year','location']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='no2', data=df)
#%%
data[['no2','type']].groupby(["type"]).median().sort_values(by='no2',ascending=False).plot.bar(color='r')
plt.show()
#%%
loc_arr=data.location.unique()
loc_arr.sort()
print (loc_arr)
print (data.head(5))

#%%

#data1=pd.DataFrame(columns=['location','type','so2','no2','rspm','spm','date','year'])
data.sort_values(by=['location'])
means = data.groupby('location').mean()
#%%
so2_mean=means.so2
no2_mean=means.no2
rspm_mean=means.rspm
spm_mean=means.spm

#%% 
print (data.columns)
print (data.shape)
#%%
df2=pd.DataFrame(columns=['location','type','so2','no2','rspm','spm','date','year'])
#%%
'''
df2=data.copy()
df2=df2.loc[data.location==loc_arr[0]]
df2.so2.fillna(so2_mean[0])
df2.no2.fillna(no2_mean[0])
df2.rspm.fillna(rspm_mean[0])
df2.spm.fillna(spm_mean[0])
'''
for i in range (0,len(loc_arr)):
    dt=data.copy()
    dt=dt.loc[data.location==loc_arr[i]]
    dt.so2.fillna(so2_mean[i],inplace=True)
    dt.no2.fillna(no2_mean[i],inplace=True)
    dt.rspm.fillna(rspm_mean[i],inplace=True)
    dt.spm.fillna(spm_mean[i],inplace=True)
    df2=df2.append(dt)
    #print (df2.shape)
#print (df2.so2)
print (df2.shape)
#print (df2.head(10))
df2.to_csv("C:/Users/SEJAL/Desktop/XL_Files/Air_quality_updated1.csv")
data=pd.read_csv("C:/Users/SEJAL/Desktop/XL_Files/Air_quality_updated1.csv")



#%%
AQI_min=0
AQI_max=500

min_so2=min(data.so2)
max_so2=max(data.so2)

min_no2=min(data.no2)
max_no2=max(data.no2)

min_rspm=min(data.rspm)
max_rspm=max(data.rspm)

min_spm=min(data.spm)
max_spm=max(data.spm)

def calculate_si(so2):
    return (((so2-min_so2)/AQI_max)*(max_so2-min_so2))
def calculate_ni(no2):
    return (((no2-min_no2)/AQI_max)*(max_no2-min_no2))
def calculate_rspmi(rspm):
    return (((rspm-min_rspm)/AQI_max)*(max_rspm-min_rspm))
def calculate_spmi(spm):
    return (((spm-min_spm)/AQI_max)*(max_spm-min_spm))

data['si']=data['so2'].apply(calculate_si)
data['ni']=data['no2'].apply(calculate_ni)
data['rspmi']=data['rspm'].apply(calculate_rspmi)
data['spmi']=data['spm'].apply(calculate_spmi)

#%%
def calculate_aqi(si,ni,spmi,rspmi):
    aqi=0
    if(si>ni and si>spmi and si>rspmi):
        aqi=si
    if(spmi>si and spmi>ni and spmi>rspmi):
        aqi=spi
    if(ni>si and ni>spmi and ni>rspmi):
        aqi=ni
    if(rspmi>si and rspmi>ni and rspmi>spmi):
        aqi=rpi
    return aqi
#data['AQI']=data[['si','ni','rspmi','spmi']].apply(calculate_aqi)
#data['AQI']=data.apply(lambda x:calculate_aqi(x['si'],x['ni'],x['spi'],x['rpi']),axis=1)
data['AQI']=data[["si","ni","rspmi","spmi"]].max(axis=1) 
 
data.to_csv("C:/Users/SEJAL/Desktop/XL_Files/Air_quality_updated2.csv")
#%%
data=pd.read_csv("C:/Users/SEJAL/Desktop/XL_Files/Air_quality_updated2.csv")
print (data.head(10))
#%%
data[['AQI','location']].groupby(["location"]).median().sort_values(by='AQI',ascending=False).plot.bar(color='r')
plt.show()
#%%
df = data[['AQI','year','location']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(15,5))
sns.pointplot(x='year', y='AQI', data=df)
#%%
data[['AQI','type']].groupby(["type"]).median().sort_values(by='AQI',ascending=False).plot.bar(color='r')
plt.show()
#%%
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
le=LabelEncoder()
data['location']=le.fit_transform(data.location)
X=data.loc[:,['so2','no2','rspm','spm','location']].values
sc=StandardScaler()
sc.fit(X)
X=sc.transform(X)
#%%
y=data.loc[:,['AQI']].values
#%%
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=7)
#%%
model=LinearRegression()
model.fit(x_train,y_train)
y_pred_train=model.predict(x_train)
print ("RMSE",np.sqrt(metrics.mean_squared_error(y_train,y_pred_train)))
print ("r2_score on Train data",metrics.r2_score(y_train,y_pred_train))
r2_score=metrics.r2_score(y_train,y_pred_train)
print ("-------------------------------------")

y_pred=model.predict(x_test)
print ("r2_score on Test data",metrics.r2_score(y_test,y_pred))
r2_score=metrics.r2_score(y_test,y_pred)
print ("mean_absoult_score",metrics.mean_squared_error(y_test,y_pred))
print ("mean_squared_log_error",metrics.mean_squared_log_error(y_test,y_pred))
print ("RMSE",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))
#%%
f, ax = plt.subplots(figsize=(5,5))
sns.heatmap(data.corr())
#%%
df = data[['rspm','year']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
#f,ax=plt.subplots(figsize=(10,8))
sns.pointplot(x='year', y='rspm', data=df)

#%%
df = data[['spm','year']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
#f,ax=plt.subplots(figsize=(10,8))
sns.pointplot(x='year', y='spm', data=df)

#%%
df = data[['so2','year']].groupby(["year"]).median().reset_index().sort_values(by='year',ascending=False)
f,ax=plt.subplots(figsize=(10,8))
sns.pointplot(x='year', y='no2', data=df)



#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

RForest=RandomForestRegressor(n_estimators=255,random_state=354,min_samples_leaf=.0001)
RForest.fit(x_train,y_train)
y_predict=RForest.predict(x_test)
from sklearn.metrics import r2_score
print("Random forest:", r2_score(y_test,y_predict))


















