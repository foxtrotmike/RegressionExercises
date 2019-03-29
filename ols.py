# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:45:33 2019
A simple demo of how square loss is sensitive to outliers
Dependency: download https://github.com/foxtrotmike/plotit
@author: afsar
"""
import numpy as np
import matplotlib.pyplot as plt
from plotit import plotit,getExamples
from sklearn.svm import LinearSVC
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import roc_auc_score as auroc
class OLS:
    def __init__(self,lambdaa=0.0):
        self.w = None
        self.lambdaa = lambdaa
    def fit(self,X,y):
        self.w = np.linalg.inv(X.T.dot(X)+lambdaa).dot((X.T).dot(y))
    def decision_function(self,X):
        return X.dot(self.w)

if __name__=='__main__':
    plt.close("all")

    lambdaa = 0.0
    X,y = getExamples()
    
    ols = OLS(lambdaa = lambdaa)
    ols.fit(X,y)
    
    plt.figure()
    e = plotit(X = X, Y = y, clf = ols.decision_function, conts =[-1,0,1])
    plt.title("OLS"+" AUC:"+"{0:.2f}".format(auroc(y,ols.decision_function(X))))
    plt.show()
    
    clf = LinearSVC(C=1e1)
    clf.fit(X,y)
    plt.figure()
    plotit(X = X, Y = y, clf = clf.decision_function, conts =[-1,0,1],extent = e)
    plt.title("SVM"+" AUC:"+"{0:.2f}".format(auroc(y,clf.decision_function(X))))
    plt.show()
    
    
    X[0]=-1000*X[0] #Let's add some fun!
    
    ols2 = OLS(lambdaa = lambdaa)
    ols2.fit(X,y)
    plt.figure()
    plotit(X = X, Y = y, clf = ols2.decision_function, conts =[-1,0,1],extent = e)
    plt.title("OLS-Data Outlier"+" AUC:"+"{0:.2f}".format(auroc(y,ols2.decision_function(X))))
    plt.show()
    
    clf.fit(X,y)
    plt.figure()
    plotit(X = X, Y = y, clf = clf.decision_function, conts =[-1,0,1],extent = e)
    plt.title("SVM-Data Outlier"+" AUC:"+"{0:.2f}".format(auroc(y,clf.decision_function(X))))
    plt.show()