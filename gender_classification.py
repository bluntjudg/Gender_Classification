# -*- coding: utf-8 -*-
"""Gender Classification.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1qtOvTn8IfIJxpI3NN4-uxHq6T7dq3cO2
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler
from sklearn.metrics import mean_absolute_error, confusion_matrix, classification_report, accuracy_score

df = pd.read_csv('gender_classification.csv')

df.head()

df.info()

df.isnull().sum()

df.duplicated().sum()

df.drop_duplicates(inplace=True)
df.shape

df.describe()

df.sample(5).T

"""Here The .T attribute transposes the DataFrame, which means it swaps the rows and columns.
After transposing, what were originally rows become columns, and what were originally columns become rows.
"""

for i in df:
    if i =='gender':
      break
    if i =='forehead_height_cm':
        plt.tick_params(rotation=90)
    plt.figure(figsize=(10,5))
    sns.countplot(x=df[i], hue=df['gender'])
    plt.title(f"Whether the Participant Had {i} or Not")

x = df[df.columns[:-1]]
y = df['gender']
le=LabelEncoder()
y = le.fit_transform(y)

scaler = MinMaxScaler()
columns = x.columns
x = pd.DataFrame(scaler.fit_transform(x))
x.columns = columns
x.head()

x_train , x_test , y_train , y_test = train_test_split(x , y , test_size=0.2 ,random_state=42 )

lr = LogisticRegression()
lr.fit(x_train,y_train)

y_pred = lr.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
lr_train_acc = round(accuracy_score(y_train,lr.predict(x_train))*100,2)
lr_test_acc = round(accuracy_score(y_test,y_pred)*100,2)

print('Accuracy = ' , lr_test_acc,' %')
print("Classification report: \n{}\n".format(classification_report(y_test, y_pred)))
sns.heatmap(cm,annot=True, fmt='d', cmap='Blues', cbar=False,)
plt.title('Logistic Regresstion Confusion Matrix')

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train,y_train)

y_pred = knn.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
knn_train_acc = round(accuracy_score(y_train,knn.predict(x_train))*100,2)
knn_test_acc = round(accuracy_score(y_test,y_pred)*100,2)
print('Accuracy = ' , knn_test_acc,' %')
print("Classification report: \n{}\n".format(classification_report(y_test, y_pred)))
sns.heatmap(cm,annot=True, fmt='d', cmap='Blues', cbar=False,)
plt.title('K-Nearest Neighbors Confusion Matrix');

model=DecisionTreeClassifier()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
model_train_acc = round(accuracy_score(y_train,model.predict(x_train))*100,2)
model_test_acc = round(accuracy_score(y_test,y_pred)*100,2)
print('Accuracy = ' , knn_test_acc,' %')
print("Classification report: \n{}\n".format(classification_report(y_test, y_pred)))
sns.heatmap(cm,annot=True, fmt='d', cmap='Blues', cbar=False,)
plt.title('Tree Confusion Matrix');

models = pd.DataFrame({
    'Model': [
        'Logistic Regression','K Nearest Neighbors','tree'
    ],
    'Training Accuracy': [
        lr_train_acc,knn_train_acc,model_train_acc
    ],
    'Model Accuracy Score': [
       lr_test_acc,knn_test_acc,model_test_acc
    ]
})
models.sort_values(by='Training Accuracy', ascending=False)

models.sort_values(by='Model Accuracy Score', ascending=False).style.background_gradient(cmap='coolwarm')

plt.figure(figsize=(12,6))

plt.plot(models['Model'] , models['Model Accuracy Score'])

plt.title("Comparison of Accuracy Values")

plt.show()

print(f'\n\nAccuracy of KNN: {knn_test_acc:.2f}\n')
print(f'Accuracy of Logistic Regression: {lr_test_acc:.2f}\n')
print(f'Accuracy of tree: {model_test_acc:.2f}\n')

if(lr_test_acc > knn_test_acc and lr_test_acc > model_test_acc):
  print('Accuracy of Logistic Regression is good than KNN and tree')
elif (lr_test_acc < knn_test_acc and knn_test_acc > model_test_acc) :
  print('Accuracy of KNN is good than Logistic Regression and tree\n')
else:
  print('Accuracy of tree is good than Logistic Regression and Knn\n')

