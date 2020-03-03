#!/usr/bin/env python
# coding: utf-8

# In[1]:


# # Configuration for JupyterLab
# %config IPCompleter.greedy=True

# Importing Required Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import NearestCentroid
from sklearn.naive_bayes import GaussianNB

#Ignore Convergence warnings
from warnings import filterwarnings
filterwarnings('ignore')


# In[2]:


# Convert the csv to a dataframe
data = pd.read_csv('results.csv', encoding='ISO-8859-1', delimiter=';')
print(data.head)


# In[3]:


#Numerical Data Analysis
print(data.describe())


# In[4]:


#Categorical Data Analysis
print(data['over_50k'].value_counts(normalize=True))
print(data['country'].value_counts(normalize=True))
print(data['education_level'].value_counts(normalize=True))
print(data['marital_status'].value_counts(normalize=True))
print(data['occupation'].value_counts(normalize=True))
print(data['race'].value_counts(normalize=True))
print(data['relationship_status'].value_counts(normalize=True))
print(data['sex'].value_counts(normalize=True))


# In[5]:


#Fix issues with United-States not converting properly
# data['country'] = data['country'].replace('United-States', 'US')


# In[6]:


#Plot graph between occupation and over_50k
sns.set(style="ticks", color_codes=True)
sns.catplot(y="occupation", x="over_50k", data=data, kind="bar");


# In[7]:


#Create dummy variables for categorical values
cat_vars=['country','education_level','marital_status','occupation','race','relationship_status','sex']
for var in cat_vars:
    cat_list='var' +'_' + var
    cat_list = pd.get_dummies(data[var], prefix=var)
    data=data.join(cat_list)
data_vars=data.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]
data = data[to_keep]


# In[8]:


#Create a corrolation matrix
corrMatrix = data.corr()
plt.matshow(data.corr())
plt.show()


# In[10]:


#Create testing/training split
os = SMOTE(random_state=0)
X = data.loc[:, data.columns != 'over_50k']
y = data.loc[:, data.columns == 'over_50k']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0, shuffle=True)


# In[11]:


# #Scale the data
# scaler = StandardScaler()
# scaler.fit(X_train)
# X_train = scaler.transform(X_train) 
# X_test = scaler.transform(X_test)
# X_train


# In[12]:


#Over-sample using smote to make up for differences in classes
smote_X, smote_y = os.fit_sample(X_train, y_train)
smote_X = pd.DataFrame(data=smote_X,columns=X_train.columns)
smote_y = pd.DataFrame(data=smote_y,columns=['over_50k'])


# In[13]:


#Determine which features to use
logreg = LogisticRegression(max_iter=2000)
# rfe = RFE(logreg, 20)
# rfe = rfe.fit(smote_X, smote_y.values.ravel())
# print(rfe.support_)
# print(rfe.ranking_)


# In[14]:


#Determine top 5 features
rfe_5 = RFE(logreg, 5)
rfe_5 = rfe_5.fit(smote_X, smote_y.values.ravel())
print(rfe_5.support_)
print(rfe_5.ranking_)


# In[15]:


#Save selected features
def select_features(rfe):
    potential_features = rfe.support_
    cols = X_train.columns
    selected = []
    for i in range(potential_features.size):
        if potential_features[i] == True:
            selected.append(cols[i])
    return selected

# selected = select_features(rfe)
selected = select_features(rfe_5)
top_5 = select_features(rfe_5)
select_smote_X = smote_X[selected]
select_smote_y = smote_y
# select_smote_X


# In[16]:


# #Plot the most important relationships
# for i in top_5:
# #     data.plot(y=i, x='over_50k', style='o')
# #     data.plot(y=i, x='over_50k', style='o',kind='bar')
# #     data.plot(y=i, x='over_50k', style='o',kind='line')
# #     data.plot(y=i, x='over_50k', style='o',kind='pie')
# # data.plot(x="over_50k", y=top_5, kind="bar")


# In[17]:


#Train Logistic Regression and test on all features
logreg_all = LogisticRegression(max_iter=2000)
logreg_all.fit(smote_X, smote_y['over_50k'])
y_pred = logreg_all.predict(X_test)
print('Accuracy of Logistic Regression on all featutres: {:.2f}'.format(logreg_all.score(X_test, y_test)))
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
print(classification_report(y_test, y_pred))


# In[18]:


#Train Logistic Regression and test on selected features
logreg_select = LogisticRegression(max_iter=2000)
logreg_select.fit(smote_X, smote_y['over_50k'])
y_pred = logreg_select.predict(X_test)
print('Accuracy of Logistic Regression on select featutres: {:.2f}'.format(logreg_select.score(X_test, y_test)))
# confusion_matrix_select = confusion_matrix(y_test, y_pred)
# print(confusion_matrix_select)
print(classification_report(y_test, y_pred))


# In[19]:


#Train Multi-layer Perceptron and test on all features
clf_all = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=0)
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Multi-layer Perceptron on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_1 = confusion_matrix(y_test, y_pred)
# print(cm_1)
print(classification_report(y_test, y_pred))


# In[20]:


#Train Multi-layer Perceptron and test on selected features
clf_select = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 2), random_state=0)
clf_select.fit(smote_X, smote_y['over_50k'])
y_pred = clf_select.predict(X_test)
print('Accuracy of Multi-layer Perceptron on select featutres: {:.2f}'.format(clf_select.score(X_test, y_test)))
# cm_2 = confusion_matrix(y_test, y_pred)
# print(cm_2)
print(classification_report(y_test, y_pred))


# In[21]:


#Train Multi-layer Perceptron with fewer layers
clf_all = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(2, 2), random_state=0)
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Multi-layer Perceptron with 2 layers on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_3 = confusion_matrix(y_test, y_pred)
# print(cm_3)
print(classification_report(y_test, y_pred))


# In[22]:


#Train Multi-layer Perceptron with more layers
clf_all = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(10, 2), random_state=0)
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Multi-layer Perceptron on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_4 = confusion_matrix(y_test, y_pred)
# print(cm_4)
print(classification_report(y_test, y_pred))


# In[23]:


#Train Multi-layer Perceptron with fewer layers
clf_all = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5, 5), random_state=0)
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Multi-layer Perceptron on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_3 = confusion_matrix(y_test, y_pred)
# print(cm_3)
print(classification_report(y_test, y_pred))


# In[24]:


#Train Support Vector Classifier and test on all features
# clf_all = svm.SVC()
# clf_all.fit(smote_X, smote_y['over_50k'])
# y_pred = clf_all.predict(X_test)
# print('Accuracy of Support Vector Classifier on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# # cm_5 = confusion_matrix(y_test, y_pred)
# # print(cm_5)
# print(classification_report(y_test, y_pred))


# In[25]:


#Train Support Vector Classifier and test on selected features
# clf_select = svm.SVC()
# clf_select.fit(smote_X, smote_y['over_50k'])
# y_pred = clf_select.predict(X_test)
# print('Accuracy of Support Vector Classifier on select featutres: {:.2f}'.format(clf_select.score(X_test, y_test)))
# # cm_6 = confusion_matrix(y_test, y_pred)
# # print(cm_6)
# print(classification_report(y_test, y_pred))


# In[26]:


#Train Decision Tree and test on all features
clf_all = tree.DecisionTreeClassifier()
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Decision Tree on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_5 = confusion_matrix(y_test, y_pred)
# print(cm_5)
print(classification_report(y_test, y_pred))


# In[27]:


#Train Decision Tree and test on selected features
clf_select = tree.DecisionTreeClassifier()
clf_select.fit(smote_X, smote_y['over_50k'])
y_pred = clf_select.predict(X_test)
print('Accuracy of Decision Tree on select featutres: {:.2f}'.format(clf_select.score(X_test, y_test)))
# cm_6 = confusion_matrix(y_test, y_pred)
# print(cm_6)
print(classification_report(y_test, y_pred))


# In[28]:


#Train Nearest Centroid Classifier and test on all features
clf_all = NearestCentroid()
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Nearest Centroid Classifier on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_5 = confusion_matrix(y_test, y_pred)
# print(cm_5)
print(classification_report(y_test, y_pred))


# In[29]:


#Train Nearest Centroid Classifier and test on selected features
clf_select = NearestCentroid()
clf_select.fit(smote_X, smote_y['over_50k'])
y_pred = clf_select.predict(X_test)
print('Accuracy of Nearest Centroid Classifier on select featutres: {:.2f}'.format(clf_select.score(X_test, y_test)))
# cm_6 = confusion_matrix(y_test, y_pred)
# print(cm_6)
print(classification_report(y_test, y_pred))


# In[30]:


#Train Naive Bayes Classifier and test on all features
clf_all = GaussianNB()
clf_all.fit(smote_X, smote_y['over_50k'])
y_pred = clf_all.predict(X_test)
print('Accuracy of Naive Bayes Classifier on all featutres: {:.2f}'.format(clf_all.score(X_test, y_test)))
# cm_7 = confusion_matrix(y_test, y_pred)
# print(cm_7)
print(classification_report(y_test, y_pred))


# In[31]:


#Train Naive Bayes Classifier and test on selected features
clf_select = GaussianNB()
clf_select.fit(smote_X, smote_y['over_50k'])
y_pred = clf_select.predict(X_test)
print('Accuracy of Naive Bayes Classifier on select featutres: {:.2f}'.format(clf_select.score(X_test, y_test)))
# cm_8 = confusion_matrix(y_test, y_pred)
# print(cm_8)
print(classification_report(y_test, y_pred))


# In[ ]:





# In[ ]:




