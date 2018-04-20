# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 12:21:18 2018

@author: ADITYA
"""

import sys 
import itertools
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import neighbors

from sklearn import cross_validation
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cross_validation import KFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split

from sklearn.decomposition import RandomizedPCA
from sklearn.svm import LinearSVC, SVC
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing


train = pd.read_csv("D:/UB SPRING/PROGRAMMING FOR ANALYTICS/NEW/train.csv")
test = pd.read_csv("D:/UB SPRING/PROGRAMMING FOR ANALYTICS/NEW/test.csv")

Y = train["label"]

X = train.drop(labels = ["label"], axis = 1)

#g= sns.countplot(Y_train)

Y.value_counts()

###data normalization.

X = X/255.0
test = test/255.0
###to check for null values
#print (X.isnull().sum())
#print (Y.isnull().sum())



X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size = 0.33, random_state =2)
####################################################
###K- nearest neighbors
####################################################
'''
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
print (clf)

Y_actual = Y_val
Y_pred = clf.predict(X_val)

print (metrics.classification_report(Y_actual, Y_pred))
print (accuracy_score(Y_actual, Y_pred))

###Validation set accuracy is obtained to be 0.9637

'''
####################################################
###Logistic Regression
####################################################
'''
LogReg = LogisticRegression()
LogReg.fit(X_train,Y_train)
####training accuracy

print (LogReg.score(X_train,Y_train))

Y_pred = LogReg.predict(X_val)
Y_actual = Y_val

print (metrics.classification_report(Y_actual, Y_pred))
print (metrics.accuracy_score(Y_actual, Y_pred))
###Logistic regression gives 0.9122 accuracy
'''
####################################################
###Classification Tree
####################################################
'''
clftree = tree.DecisionTreeClassifier()
clftreefit = clftree.fit(X_train,Y_train)
Y_pred = clftreefit.predict(X_val)

print(clftreefit.score(X_train,Y_train))

Y_actual = Y_val

print (metrics.classification_report(Y_actual, Y_pred))
print (metrics.accuracy_score(Y_actual, Y_pred))

####Classification tree gives 0.8488 accuracy
'''
####################################################
###Random Forest
####################################################

clf = RandomForestClassifier(n_estimators = 100, max_depth=None, random_state=0)
clf.fit(X_train,Y_train)
Y_pred = clf.predict(X_val)

Y_actual = Y_val

print (metrics.classification_report(Y_actual, Y_pred))
print (metrics.accuracy_score(Y_actual, Y_pred))

###Random Forest gives an accuracy of 0.9619