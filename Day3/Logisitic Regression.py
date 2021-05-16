# -*- coding: utf-8 -*-
"""
Created on Fri May 14 11:18:58 2021

@author: aksha
"""


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import linear_model
import sklearn as sklearn
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

clf = linear_model.LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='newton-cg', max_iter=1000, multi_class='ovr', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None).fit(features, labels)
print(clf.score(features, labels))


df = pd.read_csv("test.csv")
index = df.pop("Id")
train_set = np.array(df)
index = np.array(index)

scalar = preprocessing.StandardScaler().fit(train_set)
scaled_train_set = scalar.transform(train_set)

y_hat = clf.predict(scaled_train_set)
y_hat = y_hat.astype(int)
index = index.astype(int)

output = np.concatenate((index.reshape(-1,1), y_hat.reshape(-1,1)), axis = 1)


outdf= pd.DataFrame(output)
outdf.to_csv("submit.csv")

