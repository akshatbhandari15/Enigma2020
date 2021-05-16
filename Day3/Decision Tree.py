# -*- coding: utf-8 -*-
"""
Created on Thu May 13 07:09:13 2021

@author: aksha
"""
# -*- coding: utf-8 -*-
"""
Created on Tue May 11 18:54:33 2021

@author: aksha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import datasets, tree, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv", index_col=(8));

city = df.pop('city')

print("NaN values in the dataset:\n" , df.isna().sum() )

sns.pairplot(df[['estimated blast radius(km)' , 'temperature(C)', 'distance from nearest carrier(nm)', 'military-to-civilian ratio', 'population density', 'relative humidity', 'squadron strength' ]], diag_kind='kde')
print(df.describe().transpose()[['mean', 'std']])

features = np.array(df)
labels = np.array(city)

for i in range(len(labels)):
    if labels[i] == 7:
        labels[i] = 0
        



scalar = preprocessing.StandardScaler().fit(features)

scaled_features = scalar.transform(features)

clf = tree.DecisionTreeClassifier()

clf = clf.fit(scaled_features, labels)

predicted = clf.predict(scaled_features)

print(f"Classification report for classifier {clf}:\n"
      f"{metrics.classification_report(labels, predicted)}\n")


disp = metrics.plot_confusion_matrix(clf, features, labels)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")

plt.show()

tree.plot_tree(clf) 


from sklearn.metrics import accuracy_score

y_predicted = clf.predict(features)
print(accuracy_score(labels, y_predicted))





