# -*- coding: utf-8 -*-
"""
Created on Sat May 15 01:36:23 2021

@author: aksha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV


df = pd.read_csv("train.csv", index_col=(4));

status_v = df.pop('Victory Status')

print("NaN values in the dataset:\n" , df.isna().sum() )

sns.pairplot(df[['Number of soldiers' , 'Number of tanks', 'Number of aircrafts' ]], diag_kind='kde')
print(df.describe().transpose()[['mean', 'std']])


features = np.array(df)
labels = np.array(status_v)

for i in range(len(labels)):
    if labels[i] == 2:
        labels[i] = 0
        
        
scaler = preprocessing.StandardScaler()
scaler.fit(features)
scaled_features = scaler.transform(features)


dtrain = xgb.DMatrix(data=scaled_features, label=labels)


params = {
    'booster': 'gbtree',
    'max_depth': 7,
    'objective': 'binary:hinge',
    'n_gpus': 0,
    'subsampling': 0.9
}
bst = xgb.train(params, dtrain)
pred_train = bst.predict(dtrain)

print(classification_report(labels, pred_train))



df = pd.read_csv("test.csv")
index = df.pop("Id")
train_set = np.array(df)
index = np.array(index)

scaler = preprocessing.StandardScaler()
scaler.fit(train_set)
scaled_train_set = scaler.transform(train_set)
dtest = xgb.DMatrix(data=scaled_train_set)
pred = bst.predict(dtest)





df = pd.read_csv("test.csv")
index = df.pop("Id")

for i in range(len(pred)):
    if pred[i] == 0:
        pred[i] = 2

train_set = np.array(df)
index = np.array(index)

y_hat = pred.astype(int)
index = index.astype(int)

output = np.concatenate((index.reshape(-1,1), y_hat.reshape(-1,1)), axis = 1)


outdf= pd.DataFrame(output)
outdf.columns = ['Id', 'Category']
outdf.to_csv("submit.csv")
