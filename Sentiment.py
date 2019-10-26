# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 00:16:14 2019

@author: Admin
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
#%%
df = pd.read_csv('C:/Users/Admin/Desktop/Project/kindle_reviews.csv', na_filter=False)
df.drop([ 'unixReviewTime', 'reviewTime', 'reviewerID', 'reviewerName'], axis=1, inplace=True)
newdf = df[:10000]
print(df.head())
print(df.columns)
print(df.dtypes)
print ("Shape of the dataset - ", newdf.shape)
#check for the missing values
newdf.apply(lambda x: sum(x.isnull()))
print(newdf['overall'].value_counts())
#newdf1 = newdf[newdf['overall'] != 3]

newdf['Positively Rated'] = np.where(newdf['overall'] > 3, 1, 0)
print(newdf['Positively Rated'].mean())

#%%
from  sklearn.model_selection import train_test_split
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(newdf['reviewText'],newdf['Positively Rated'], random_state=214)
print('X_train first entry: ', X_train.iloc[1])
print('\nX_train shape: ', X_train.shape)


#%%

from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import roc_auc_score


# Fit the CountVectorizer to the training data
vect = CountVectorizer().fit(X_train)
# transform the documents in the training data to a document-term matrix
X_train_vectorized = vect.transform(X_train)
# Train the model
model = LogisticRegression()
model.fit(X_train_vectorized, y_train)
# Predict the transformed test documents
predictions = model.predict(vect.transform(X_test))
print('AUC: ', roc_auc_score(y_test, predictions))
# get the feature names as numpy array
feature_names = np.array(vect.get_feature_names())
# Sort the coefficients from the model
sorted_coef_index = model.coef_[0].argsort()
# Find the 10 smallest and 10 largest coefficients
# The 10 largest coefficients are being indexed using [:-11:-1] 
# so the list returned is in order of largest to smallest
pop=('Smallest Coefs:\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
topp=('Largest Coefs: \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))

#%%
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report

#Model Evaluation
acc = accuracy_score(y_test, predictions, normalize=True)
hit = precision_score(y_test, predictions, average=None)
capture = recall_score(y_test, predictions, average=None)
print('Model Accuracy:%.2f'%acc)
print(classification_report(y_test, predictions))



#%%

X = np.concatenate((X_train, X_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)
print("Classes: ")
print(np.unique(y))
#%%
from wordcloud import WordCloud

wordcloud = WordCloud(
                          background_color='black',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(pop)

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

wordcloud = WordCloud(
                          background_color='white',
                          max_words=200,
                          max_font_size=40, 
                          random_state=42
                         ).generate(topp)

print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()

print('Number of books in data:',newdf['asin'].nunique())
plt.figure(figsize=(14,10))
cnt = df['asin'].value_counts().to_frame()[0:20]
#plt.xscale('log')
sns.barplot(x= cnt['asin'], y =cnt.index, data=cnt, palette='ocean',orient='h')
#plt.title('Distribution of Wine Reviews by Top 20 Countrie');



f, ax = plt.subplots(1,2,figsize=(6,3))
ax1,ax2 = ax.flatten()
sns.distplot(df['overall'].fillna(newdf['overall'].mean()),color='r',ax=ax1)
ax1.set_title('Distrbution of rates')
sns.boxplot(x = newdf['overall'], ax=ax2)
ax2.set_ylabel('')
ax2.set_title('Boxplot of rates')

import squarify
cnt = newdf.groupby(['asin',])['overall'].mean().sort_values(ascending=False).to_frame()
plt.figure(figsize=(12,8))
squarify.plot(cnt['overall'].fillna(0.1),color=sns.color_palette('rainbow'),label=cnt.index)
#%% 1

#Extra Trees Classifeir
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import LabelEncoder

lb_en=LabelEncoder()
newdf['reviewText']=lb_en.fit_transform(newdf['reviewText'])
newdf['reviewText'].unique()
print(newdf['reviewText'].head())
#print(newdf['reviewText'].shape)

ar=newdf['reviewText']
z=ar.values.reshape(-1,1)

X_train, X_test, y_train, y_test = train_test_split(z,newdf['Positively Rated'], random_state=241)

et_clf = ExtraTreesClassifier(n_estimators=250)
et_clf.fit(X_train,y_train)
kfold = StratifiedKFold(n_splits=7,random_state=7)
results= cross_val_score(et_clf,X_train,y_train,cv=kfold)
print("\n Extra Tree..........")
#print("Train - Accuracy:", results.mean())
y_train_pred=et_clf.predict(X_train)
print("Train - Accuracy:", accuracy_score(y_train,y_train_pred))

y_test_pred=et_clf.predict(X_test)
print("Test - Accuracy:", accuracy_score(y_test,y_test_pred))

#%% 2

from sklearn.tree import DecisionTreeClassifier
dt_clf=DecisionTreeClassifier(max_depth=None,criterion='gini',random_state=241,min_samples_leaf=.0001)
dt_clf.fit(X_train,y_train)
kfold = StratifiedKFold(n_splits=10,random_state=7)
results= cross_val_score(et_clf,X_train,y_train,cv=kfold)
print("\n Decision Tree..........")
#print("Train - Accuracy:", results.mean())
y_train_pred=dt_clf.predict(X_train)
print("Train - Accuracy:", accuracy_score(y_train,y_train_pred))

y_test_pred=dt_clf.predict(X_test)
print("Test - Accuracy:", accuracy_score(y_test,y_test_pred))

#%% 3

#Bagging Classifier
from sklearn.ensemble import BaggingClassifier
dt_clf_bag = BaggingClassifier(base_estimator = dt_clf,n_estimators=10,random_state=3)
dt_clf_bag.fit(X_train,y_train)
results= cross_val_score(dt_clf_bag,X_train,y_train,cv=kfold)
print("\n Decision Tree  with Bagging..........")
#print("Train - Accuracy:", results.mean())

y_train_pred=dt_clf_bag.predict(X_train)
print("Train - Accuracy:", accuracy_score(y_train,y_train_pred))
y_test_pred=dt_clf_bag.predict(X_test)
print("Test - Accuracy:", accuracy_score(y_test,y_test_pred))

#%% 4
#Gradient Boosting
from sklearn.ensemble import GradientBoostingClassifier
gbt_clf =  GradientBoostingClassifier(n_estimators=15,learning_rate=0.1,random_state=3 )
gbt_clf.fit(X_train,y_train)
results = cross_val_score(gbt_clf,X_train,y_train,cv=kfold)
#print("\n Gradent Boosting - CV Train : %.2f" %results.mean())
y_train_pred = gbt_clf.predict(X_train)
#print("\n Gradent Boosting - Train : %.2f" % accuracy_score(y_train,y_train_pred))
y_test_pred = gbt_clf.predict(X_test)
print("\n Gradent Boosting - Test : %.2f" % accuracy_score(y_test,y_test_pred))

#%% 5
import xgboost as xgb
from xgboost.sklearn import XGBClassifier

num_trees=100
num_folds=5
kfold = StratifiedKFold(n_splits=10,random_state=7)
xgb_clf = XGBClassifier(n_estimators = 100,objective='binary:logistic',seed=5)

xgb_clf.fit(X_train,y_train,early_stopping_rounds=10,eval_set=[(X_test,y_test)],verbose=1)

results = cross_val_score(xgb_clf,X_train,y_train,cv=kfold)
print("\nXGBoost - CV Train : %.2f" %results.mean())
y_train_pred = xgb_clf.predict(X_train)
print("\n XGBoost - Train : %.2f" % accuracy_score(y_train,y_train_pred))
y_test_pred = xgb_clf.predict(X_test)
print("\n XGBBoost - Test : %.2f" % accuracy_score(y_test,y_test_pred))

xgb.plot_importance(xgb_clf)


