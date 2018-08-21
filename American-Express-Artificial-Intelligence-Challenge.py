# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:18:06 2018

@author: suman
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 00:04:38 2018

@author: suman
"""

import numpy as np
import pandas as pd


from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import cross_val_score


from sklearn.metrics import accuracy_score
from sklearn.grid_search import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt

train=pd.read_excel("F:/american express dataset/Dataset - Problem 2/trainnew.xlsx", index_col=None)
test=pd.read_excel("F:/american express dataset/Dataset - Problem 2/testnew.xlsx", index_col=None)

feature=train.pop('label')

index_test=test['key']
"""
train.drop(['V2','V4','V6','V7','V8','V9','V10','V15','V19','V26','V27','V29','V42','V44','V46','V50','V51','V52','V53','V54'],axis=1)
test.drop(['V2','V4','V6','V7','V8','V9','V10','V15','V19','V26','V27','V29','V42','V44','V46','V50','V51','V52','V53','V54'],axis=1)
"""

scaler=StandardScaler()

train[['V1','V3','V5']]=scaler.fit_transform(train[['V1','V3','V5']])
test[['V1','V3','V5']]=scaler.fit_transform(test[['V1','V3','V5']])


"""
modelCV=LGBMClassifier(num_leaves=5500, min_data_in_leaf=17, max_bin=800, bagging_fraction=0.74, max_depth=50,objective='binary')
rfe = RFE(modelCV)
rfe = rfe.fit(train,feature)
print(rfe.support_)
print(rfe.ranking_)
"""
#print(train.head())
#print(test.head())

"""
cols=['key','V1','V2','V3','V4','V6','V7','V8','V10','V11','V13','V14','V26','V36','V37','V43','V52']

train=train[cols]
test=test[cols]
"""

seed = 7
scoring = 'accuracy'
kfold = model_selection.KFold(n_splits=2, random_state=7)

"""
params={ 'learning_rate':0.01,
 'n_estimator':5000,
 'max_dept':8,
 'min_child_weigh':6,
 'gamma':0,
 'subsample':0.8,
 'colsample_bytree':0.8,
 'reg_alpha':0.005,
 'nthread':4,
 'scale_pos_weight':1,
 'seed':27 }
"""



"""
modelCV = XGBClassifier(n_estimators=700,
 max_depth=8,
 min_child_weight=7,
 gamma=0,
 subsample=0.8,
 objective='binary:logistic',
 colsample_bytree=0.8,
 nthread=4,
 scale_pos_weight=1,
 reg_alpha=0.005)
"""

scoring = 'accuracy'
modelCV=LGBMClassifier(num_leaves=10000,num_iterations=800,min_data_in_leaf=17, max_bin=800, bagging_fraction=0.74, max_depth=50,objective='binary')
"""
param_grid={:[100,250,500,750,1000,2500,5500,7500,10000]}
grid = GridSearchCV(modelCV,param_grid)
grid.fit(train, feature)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)
"""
"""
results = model_selection.cross_val_score(modelCV, train, feature, cv=kfold, scoring=scoring)
print(results.mean())
"""


modelCV.fit(train,feature)
results = modelCV.predict_proba(test)


my_submission = pd.DataFrame({'key': index_test, 'score': results[:,1]})
my_submission.to_csv('F:/american express dataset/Dataset - Problem 2/americanLGBM62.csv', index=False)

