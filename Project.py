# -*- coding: utf-8 -*-
"""Created on Fri Aug 16 14:35:50 2019"
@author: Aakanksha Mahajan
"""
#%%
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score  
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#Loading dataset
wine = pd.read_csv('D:/Project/winequality-red.csv')
#Making binary classificaion for the response variable.
#Dividing wine as good and bad by giving the limit for the quality
bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()
#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])
wine['quality'].value_counts()
sns.countplot(wine['quality'])
#Now seperate the dataset as response variable and feature variabes
X = wine.drop('quality', axis = 1)
y = wine['quality']
#Train and Test splitting of data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
#Applying Standard scaling to get optimized result
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#Let's see how our model performed
print("For Random Forest"+ str(classification_report(y_test, pred_rfc)))
print("For Random Forest"+ str(confusion_matrix(y_test, pred_rfc)))
lr = LogisticRegression()
lr.fit(X_train, y_train)
pred_lr = lr.predict(X_test)
print("For Logistic Regression"+ str(classification_report(y_test, pred_lr)))
print("For Logistic Regression"+ str(confusion_matrix(y_test, pred_lr)))
print(y_test)

pipeline=Pipeline([('rfc',RandomForestClassifier(criterion='entropy',random_state=50))])
parameters={
    'rfc__n_estimators':(25,50,100),
    'rfc__max_depth':(5,10,15),
    'rfc__min_samples_split':(2,3),
    'rfc__min_samples_leaf':(1,2),
}
grid_search=GridSearchCV(pipeline,parameters,cv=5,scoring='accuracy')
grid_search.fit(X_train,y_train)
print('Best score: %0.3f'%grid_search.best_score_)
best_parameters=grid_search.best_estimator_.get_params()
for param in sorted(parameters.keys()):
    print('\t%s: %r' % (param,best_parameters[param]))
y_test_pred=grid_search.predict(X_test)
print('Test-Accuracy:'+str(accuracy_score(y_test,y_test_pred)))

