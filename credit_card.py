import os
import numpy as np
import pandas as pd

os.chdir('C:/Users/Samruddhi/Desktop/Fraud detection')  # Setting directory
data= pd.read_csv('creditcard.csv')  # Importing data

print('Class Count:',data['Class'].value_counts())  # Target Class count

# Feature Scaling
from sklearn.preprocessing import StandardScaler

data['norm_Amount']= StandardScaler().fit_transform(np.array(data['Amount']).reshape(-1,1))  # Rescaling data between -1to 1
data['norm_time']= StandardScaler().fit_transform(np.array(data['Time']).reshape(-1,1))
data=data.drop(['Time','Amount'], axis=1) # Dropping features
data= data.iloc[:,np.r_[0:28,29,30,28]]   # Feature rearrange

from sklearn.model_selection import train_test_split
X= data.iloc[:,0:30]
y= data.iloc[:,30]
X_train, X_test, y_train, y_test= train_test_split(X,y,test_size= 0.2, random_state=0) # Data split in train and test sets

# Using SMOTE for oversampling minority class
from imblearn.over_sampling import SMOTE
sm= SMOTE(random_state=2)
X_train_new, y_train_new = sm.fit_sample(X_train, y_train.ravel())  # Over-sampling minority class using SMOTE

print('Before Over-sampling, count of label "1": {}'.format(sum(y_train ==0)))
print('Before Over-sampling, count of label "0": {}'.format(sum(y_train ==1)))

print('After Over-sampling, count of label "1": {}'.format(sum(y_train_new == 0)))
print('After Over-sampling, count of label "0": {}'.format(sum(y_train_new ==1)))

print(len(y_train_new))
print(len(X_train_new))
new= X_train_new.copy()
new['Class']= np.nan
for i in range(0,len(X_train_new['norm_Amount'])):   # Arranging separate data in a single dataframe
    new['Class'][i]= y_train_new[i]
print(new.head())
new['Class']= new['Class'].astype(int)
# new.to_csv('new_data.csv', index= False)

data= pd.read_csv('new_data.csv')
X= data.iloc[:,0:30]
y= data.iloc[:,30]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.25)  # Using Stratify sampling for cross validation

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# K-Fold Cross Validation
from sklearn import metrics
from sklearn.model_selection import KFold

kfold= KFold(n_splits=2)  # Using K-fold cross validation

from sklearn.model_selection import cross_val_score
# Logistic Regression
from sklearn.linear_model import LogisticRegression
LR= LogisticRegression()

acc= cross_val_score(LR, X_train, y_train, cv=kfold, scoring='accuracy').mean()
pre= cross_val_score(LR, X_train, y_train, cv=kfold, scoring='precision').mean()
rec= cross_val_score(LR, X_train, y_train, cv=kfold, scoring='recall').mean()
fs= cross_val_score(LR, X_train, y_train, cv=kfold, scoring='f1').mean()
print('Logistic Regression:')
print('Accuracy: ',acc,' Precision: ', pre,' Recall: ', rec, 'F1_score:',fs)

# KNN
from sklearn.neighbors import KNeighborsClassifier
KNN= KNeighborsClassifier()

acc= cross_val_score(KNN, X_train, y_train, cv=kfold, scoring='accuracy').mean()
pre= cross_val_score(KNN, X_train, y_train, cv=kfold, scoring='precision').mean()
rec= cross_val_score(KNN, X_train, y_train, cv=kfold, scoring='recall').mean()
fs= cross_val_score(KNN, X_train, y_train, cv=kfold, scoring='f1').mean()
print('KNN:')
print('Accuracy: ',acc,' Precision: ', pre,' Recall: ', rec, 'F1_score:',fs)

# SVM
from sklearn.svm import SVC
SVM= SVC()

acc= cross_val_score(SVM, X_train, y_train, cv=kfold, scoring='accuracy').mean()
pre= cross_val_score(SVM, X_train, y_train, cv=kfold, scoring='precision').mean()
rec= cross_val_score(SVM, X_train, y_train, cv=kfold, scoring='recall').mean()
fs= cross_val_score(SVM, X_train, y_train, cv=kfold, scoring='f1').mean()
print('SVM:')
print('Accuracy: ',acc,' Precision: ', pre,' Recall: ', rec, 'F1_score:',fs)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
NB= GaussianNB()

acc= cross_val_score(NB, X_train, y_train, cv=kfold, scoring='accuracy').mean()
pre= cross_val_score(NB, X_train, y_train, cv=kfold, scoring='precision').mean()
rec= cross_val_score(NB, X_train, y_train, cv=kfold, scoring='recall').mean()
fs= cross_val_score(NB, X_train, y_train, cv=kfold, scoring='f1').mean()
print('Naive Bayes:')
print('Accuracy: ',acc,' Precision: ', pre,' Recall: ', rec, 'F1_score:',fs)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier()

acc= cross_val_score(RF, X_train, y_train, cv=kfold, scoring='accuracy').mean()
pre= cross_val_score(RF, X_train, y_train, cv=kfold, scoring='precision').mean()
rec= cross_val_score(RF, X_train, y_train, cv=kfold, scoring='recall').mean()
fs= cross_val_score(RF, X_train, y_train, cv=kfold, scoring='f1').mean()
print('Random Forest:')
print('Accuracy: ',acc,' Precision: ', pre,' Recall: ', rec, 'F1_score:',fs)

# On the basis of F1 score we select Random Forest(maximum F1 score among others) for the Model Development
# Random Forest: Accuracy:  0.9558823529411764  Precision:  0.9883112854262226  Recall:  0.9211207888880362 F1_score: 0.9538007052196982
# Model Development
model= RandomForestClassifier().fit(X,y)
print('Random Forest model built successfully!')