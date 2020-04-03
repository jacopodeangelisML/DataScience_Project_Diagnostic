# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 11:57:11 2020

@author: Iacopo
"""

##IMPORT PACKAGES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

##LOADING DATASET
df = pd.read_csv('Toddler Autism dataset July 2018.csv')

##EXPLORING DATASET
df.info()
df.head()
df.isnull().count()
des = df.describe()

##SELECTING RELEVANT FEATURES AND OUTCOME
y = df.iloc[:,18]
y = pd.get_dummies(y)
y = y['Yes']
df = df.iloc[:,11:]
df = df.drop('Who completed the test', axis = 1)
df = df.drop('Qchat-10-Score', axis = 1)
X = df.iloc[:,0:5]

##ANALYZING FEATURES'LEVELS DISTRIBUTION
X['Ethnicity'].value_counts()
X['Sex'].value_counts()
X['Jaundice'].value_counts()
X['Family_mem_with_ASD'].value_counts()


##DEALING WITH CATEGORICAL DATA
gender = pd.get_dummies(X['Sex'],drop_first = True)
ethnicity = pd.get_dummies(X['Ethnicity'], drop_first = True)
jaundice = pd.get_dummies(X['Jaundice'],drop_first = True)
familiarity = pd.get_dummies(X['Family_mem_with_ASD'], drop_first = True)

X = pd.concat([gender,ethnicity,jaundice,familiarity],axis = 1)
X['Jaundice'] = X.iloc[:,11]
X['familiarity'] = X.iloc[:,12]
X = X.drop('yes', axis = 1)
X['male'] = X.iloc[:,0]
X = X.drop('m', axis = 1)

##SPLITTING INTO TRAINING SET AND TEST SET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=42)

## DECISION TREE ALGHORITM

#Training the model
from sklearn.tree import DecisionTreeClassifier
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train,y_train)

#Predictions
y_pred = Dtree.predict(X_test)

#Model Evaluation: Confusion Matrix, Precision, Recall, F1 score, AUC
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(roc_auc_score(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, y_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Applying k-Fold Cross Validation (k = 10)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = Dtree, X = X_train, y = y_train, cv = 10)

#Model evaluation after K-Fold Cross Validation: Precision, Recall(sensitivity), f1 score, AUC
accuracies.mean()
accuracies.std()
Pscores= cross_val_score(estimator = Dtree, X = X_train, y = y_train, cv=10, scoring='precision_weighted')
precision=round((Pscores.mean()*100),3)
recall = cross_val_score(estimator = Dtree, X = X_train, y = y_train, cv=10, scoring='recall_weighted')
sensitivity=round((recall.mean()*100),3)
y_train1 = 1-y_train #for Specificity
recall = cross_val_score(estimator = Dtree, X = X_train, y = y_train1, cv=10, scoring='recall_weighted')#for specificity
specificity=round((recall.mean()*100),3)

from sklearn.model_selection import StratifiedKFold
from scipy import interp
cv = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_train, y_train):
    probas_ = Dtree.fit(X_train,y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=10)
plt.ylabel('True Positive Rate',fontsize=10)
plt.title('Cross-Validation ROC of SVM',fontsize=10)
plt.legend(loc="lower right", prop={'size': 10})
plt.show()

##RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)

#Model Evaluation: Confusion Matrix, Precision, Recall, F1 score, AUC
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc
print(confusion_matrix(y_test,rfc_pred))
print(classification_report(y_test,rfc_pred))
print(roc_auc_score(y_test, rfc_pred))
fpr, tpr, _ = roc_curve(y_test, rfc_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Applying k-Fold Cross Validation (k = 10)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv = 10)

#Model evaluation after K-Fold Cross Validation: Precision, Recall(sensitivity), f1 score, AUC
accuracies.mean()
accuracies.std()
Pscores= cross_val_score(estimator = rfc, X = X_train, y = y_train, cv=10, scoring='precision_weighted')
precision=round((Pscores.mean()*100),3)
recall = cross_val_score(estimator = rfc, X = X_train, y = y_train, cv=10, scoring='recall_weighted')
sensitivity=round((recall.mean()*100),3)
y_train1 = 1-y_train #for Specificity
recall = cross_val_score(estimator = rfc, X = X_train, y = y_train1, cv=10, scoring='recall_weighted')#for specificity
specificity=round((recall.mean()*100),3)

from sklearn.model_selection import StratifiedKFold
from scipy import interp
cv = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_train, y_train):
    probas_ = rfc.fit(X_train,y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=10)
plt.ylabel('True Positive Rate',fontsize=10)
plt.title('Cross-Validation ROC of SVM',fontsize=10)
plt.legend(loc="lower right", prop={'size': 10})
plt.show()

##GAUSSIAN NAIVE BAYES
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train, y_train)
gnb_pred = rfc.predict(X_test)

#Model Evaluation: Confusion Matrix, Precision, Recall, F1 score, AUC
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score, auc
print(confusion_matrix(y_test,gnb_pred))
print(classification_report(y_test,gnb_pred))
print(roc_auc_score(y_test, gnb_pred))
fpr, tpr, _ = roc_curve(y_test, gnb_pred)
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)')
plt.plot([0, 1], [0, 1], 'k--')  
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.show()

#Applying k-Fold Cross Validation (k = 10)
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = gnb, X = X_train, y = y_train, cv = 10)

#Model evaluation after K-Fold Cross Validation: Precision, Recall(sensitivity), f1 score, AUC
accuracies.mean()
accuracies.std()
Pscores= cross_val_score(estimator = gnb, X = X_train, y = y_train, cv=10, scoring='precision_weighted')
precision=round((Pscores.mean()*100),3)
recall = cross_val_score(estimator = gnb, X = X_train, y = y_train, cv=10, scoring='recall_weighted')
sensitivity=round((recall.mean()*100),3)
y_train1 = 1-y_train #for Specificity
recall = cross_val_score(estimator = gnb, X = X_train, y = y_train1, cv=10, scoring='recall_weighted')#for specificity
specificity=round((recall.mean()*100),3)

from sklearn.model_selection import StratifiedKFold
from scipy import interp
cv = StratifiedKFold(n_splits=10)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
plt.figure(figsize=(10,10))
i = 0
for train, test in cv.split(X_train, y_train):
    probas_ = gnb.fit(X_train,y_train).predict_proba(X_test)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    plt.plot(fpr, tpr, lw=1, alpha=0.3,
             label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))

    i += 1
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
         label='Chance', alpha=.8)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                 label=r'$\pm$ 1 std. dev.')

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel('False Positive Rate',fontsize=10)
plt.ylabel('True Positive Rate',fontsize=10)
plt.title('Cross-Validation ROC of SVM',fontsize=10)
plt.legend(loc="lower right", prop={'size': 10})
plt.show()