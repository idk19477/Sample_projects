#%%
import pandas as pd
import matplotlib as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")#, category=DeprecationWarning)

df=pd.read_csv(r"C:\Users\Admin\Desktop\dataset\indian_liver_patient.csv")
print(df.columns)
df.info()
df['Dataset']=df['Dataset'].replace(2,0)
df['Gender'] = df['Gender'].replace('Male', 1)
df['Gender'] = df['Gender'].replace('Female', 0)
print(df.head())
#%%
df.isnull().any()
df['Albumin_and_Globulin_Ratio'].isnull().sum()
#%%

missing_values_rows = df[df.isnull().any(axis=1)]
print(missing_values_rows)

df['Albumin_and_Globulin_Ratio'] = df.fillna(df['Albumin_and_Globulin_Ratio'].median())
df.isnull().any()

#%%
#df.hist()
print(df['Albumin_and_Globulin_Ratio'].median())
#print(df.describe())
#%%
sns.countplot(data=df, x = 'Dataset', label='Count')

LD, NLD = df['Dataset'].value_counts()
print('Number of patients diagnosed: ',LD)
print('Number of patients not diagnosed: ',NLD)
#%%
#male and female

sns.countplot(data=df, x = 'Gender', label='Count')
male, female = df['Gender'].value_counts()
print('Number of Males:', male)
print('Number of Females:', female)
#%%
#sns.factorplot(x="Age", y="Gender", hue="Dataset", data=df)

#df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).count().sort_values(by='Dataset', ascending=False)


df[['Gender', 'Dataset','Age']].groupby(['Dataset','Gender'], as_index=False).mean().sort_values(by='Dataset', ascending=False)

#%%
corr = df.corr()
print(corr)
import matplotlib.pyplot as plt

plt.figure(figsize= (16, 10))
sns.heatmap(df.corr(), annot=True)
plt.show()

#%%

g = sns.PairGrid(df, hue = "Dataset", vars=['Age','Total_Bilirubin','Total_Protiens'])
g.map(plt.scatter)
plt.show()

#%%
from sklearn.model_selection import train_test_split
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=3,test_size=0.20)

#%%

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

class_logis = LogisticRegression(random_state=3)

class_logis.fit(X_train,y_train)

y_pred_logis = class_logis.predict(X_test)

cm_logis = confusion_matrix(y_test,y_pred_logis)
print(cm_logis)

accuracy_logis = accuracy_score(y_test,y_pred_logis)
print('The accuracy of LogisticRegression is : ', str(accuracy_logis*100) , '%')

#%%
from sklearn.svm import SVC

class_svc = SVC(kernel='rbf', random_state=3, gamma='auto', probability = True)

class_svc.fit(X_train,y_train)

y_pred_svc = class_svc.predict(X_test)

cm_svc = confusion_matrix(y_test,y_pred_svc)
print(cm_svc)

accuracy_svc = accuracy_score(y_test,y_pred_svc)
print('The accuracy of SupportVectorClassification is : ', str(accuracy_svc*100) , '%')

#%%
from sklearn.ensemble import RandomForestClassifier

class_rfc = RandomForestClassifier(n_estimators=200, criterion='entropy',random_state=3 )

class_rfc.fit(X_train,y_train)

y_pred_rfc = class_rfc.predict(X_test)

cm_rfc = confusion_matrix(y_test,y_pred_rfc)
print(cm_rfc)

accuracy_rfc = accuracy_score(y_test,y_pred_rfc)
print('The accuracy of RandomForestClassifier is : ', str(accuracy_rfc*100) , '%')
#%%
from sklearn.ensemble import BaggingClassifier

bag_clf_r = BaggingClassifier(
        RandomForestClassifier(), n_estimators = 500, max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf_r.fit(X_train, y_train)

y_pred = bag_clf_r.predict(X_test)
accuracy_score(y_test, y_pred)
accuracy_bag = accuracy_score(y_test, y_pred)
print(accuracy_bag)
#%%
from sklearn.ensemble import BaggingClassifier

bag_clf = BaggingClassifier(
        SVC(), n_estimators = 500, max_samples=100, bootstrap=True, n_jobs=-1)
bag_clf.fit(X_train, y_train)

y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)

#%%
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
        LogisticRegression(), n_estimators = 100, algorithm="SAMME.R", learning_rate=0.2)
ada_clf.fit(X_train, y_train)

y_pred = ada_clf.predict(X_test)
accuracy_ada = accuracy_score(y_test, y_pred)
print(accuracy_ada*100)



#%%
models_comparison = [['Logistic Regression',accuracy_logis*100],
                     ['Support Vector Classfication',accuracy_svc*100], 
                     ['Random Forest Classifiaction',accuracy_rfc*100],
                     ['Adaboost Classifier Logistic', accuracy_ada*100],
                     ['Bagging Random Forest', accuracy_bag*100]
                    ]
models_comparison_df = pd.DataFrame(models_comparison,columns=['Model','% Accuracy'])
models_comparison_df.head()

#%%
fig = plt.figure(figsize=(10,4))
sns.set()
sns.barplot(x='Model',y='% Accuracy',data=models_comparison_df,palette='Dark2')
plt.xticks(size=10)
plt.ylabel('% Accuracy',size=4)
plt.xlabel('Model',size=4)
























