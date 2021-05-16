# -*- coding: utf-8 -*-
"""
Created on Fri May 14 14:44:37 2021

@author: aksha
"""

#XGBoost tuning

# -*- coding: utf-8 -*-
"""
Created on Fri May 14 12:16:43 2021

@author: aksha
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing
from sklearn import metrics
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

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
        
        


dtrain = xgb.DMatrix(data=features, label=labels)


params = {
    'booster': 'dart',
    'max_depth': 5,
    'objective': 'multi:softmax',
    'num_class': 7,
    'n_gpus': 0,
    'subsample': 0.8,
    'lambda': 1,
    'sample_type': 'uniform',
    'rate_drop': 0,
    'normalize_type': 'forest',
    'gamma': 0,
    'min_child_weight': 0.5
}
bst = xgb.train(params, dtrain)


df = pd.read_csv("test.csv")
index = df.pop("Id")
train_set = np.array(df)
index = np.array(index)



dtest = xgb.DMatrix(data=train_set)


pred_train = bst.predict(dtrain)

print(classification_report(labels, pred_train))

pred = bst.predict(dtest, ntree_limit=15)

df = pd.read_csv("test.csv")
index = df.pop("Id")

for i in range(len(pred)):
    if pred[i] == 0:
        pred[i] = 7

train_set = np.array(df)
index = np.array(index)

y_hat = pred.astype(int)
index = index.astype(int)

output = np.concatenate((index.reshape(-1,1), y_hat.reshape(-1,1)), axis = 1)


outdf= pd.DataFrame(output)
outdf.columns = ["Id", "city"]
outdf.to_csv("submit.csv")













