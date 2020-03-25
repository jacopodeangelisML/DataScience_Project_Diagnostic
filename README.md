# Identifying Mammographic Mass Benignancy vs Malignancy through Decision Tree and Random Forest algorithms
#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load and explore dataset
data = pd.read_csv('mammographic_masses_data_clean.csv')
data.head()
data.info()
data.describe()

#Checking missing values
pd.isna(data).sum()

#Data overview
data['Target'].value_counts()
sns.countplot(data['Target'])
sns.distplot(data['Age'], kde = False, bins = 100)
sns.factorplot(x = 'Target', y = 'Age', data = data, kind = 'box')
sns.countplot(x = 'Shape', hue = 'Target', data = data)
sns.countplot(x = 'Margin', hue = 'Target', data = data)
sns.countplot(x = 'Density', hue = 'Target', data = data)

#Dropping BI-RADS that is non-predictive
data = data.drop('BI-RADS', axis = 1)

#Training and Test set Splitting
from sklearn.model_selection import train_test_split
X = data.drop('Target', axis = 1)
y = data['Target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30,random_state=42)

#Training the model
from sklearn.tree import DecisionTreeClassifier
Dtree = DecisionTreeClassifier()
Dtree.fit(X_train,y_train)

#Predictions
y_pred = Dtree.predict(X_test)

#Model Evaluation
from sklearn.metrics import classification_report,confusion_matrix, roc_curve, roc_auc_score
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


#RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
print(classification_report(y_test,rfc_pred))
print(confusion_matrix(y_test,rfc_pred))

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
