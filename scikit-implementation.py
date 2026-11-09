# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 20:39:23 2026

@author: USER
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import log_loss

df=pd.read_csv('./processed.csv')
columns=["hour","month","day","location_longitude","location_latitude"]

model=LogisticRegression(solver="newton-cholesky",
    C=1,
    fit_intercept=True,
    max_iter=1000,
    tol=1e-4)

model.fit(df[columns],df['rained'])
print("Intercept:", model.intercept_)
print("Weights:", model.coef_)
print("Iterations:", model.n_iter_)
prediction=model.predict(df[columns])
prob=model.predict_proba(df[columns])
confusion=confusion_matrix(df['rained'], prediction, labels=[0,1] )

lloss=log_loss(df['rained'],prob,labels=[0,1])
l2_loss=np.sum(model.coef_** 2)/(2*1*len(df))
total_loss=lloss+l2_loss
print("total loss:", total_loss)

test_df=pd.read_csv('./test.csv')
columns=["hour","month","day","location_longitude","location_latitude"]
prediction=model.predict(test_df[columns])
prob=model.predict_proba(test_df[columns])
confusion=confusion_matrix(test_df['rained'], prediction, labels=[0,1] )