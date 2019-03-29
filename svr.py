# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 11:10:23 2019
A comparison of OLS and SVR over a toy example
try changing the gamma and other parameters
@author: afsar
"""

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as OLS
from sklearn.metrics import r2_score as metric
from sklearn.metrics import mean_squared_error as metric
from sklearn.metrics import mean_absolute_error as metric
from scipy.stats import pearsonr 
def metric(y,z):
    return pearsonr(y,z)[0]
# #############################################################################
# Generate sample data
X = np.sort(5 * np.random.rand(40, 1), axis=0)
y = 3-np.sin(X).ravel()

# #############################################################################
# Add noise to targets
y[::5] += 3 * (0.5 - np.random.rand(8))

# #############################################################################
# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1000, gamma='auto', epsilon=.1)
svr_lin = SVR(kernel='linear', C=100, epsilon=.1)
svr_poly = SVR(kernel='poly', C=100, gamma='auto', degree=2, epsilon=.1, coef0=1)
svr_poly = OLS()
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

# #############################################################################
# Look at the results
lw = 2

svrs = [svr_poly, svr_lin, svr_rbf]
kernel_label = ['OLS', 'Linear','RBF' ]
model_color = ['m', 'c', 'g']
plt.close("all")
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    axes[ix].plot(X, svr.fit(X, y).predict(X), color=model_color[ix], lw=lw,
                  label='{} model'.format(kernel_label[ix]))
    axes[ix].scatter(X,
                 y,
                 facecolor="none", edgecolor="k", s=50,
                 label='other training data')
    try:
        axes[ix].scatter(X[svr.support_], y[svr.support_], facecolor="none",
                     edgecolor=model_color[ix], s=50,
                     label='{} support vectors'.format(kernel_label[ix]))
    except:
        pass
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].grid()

fig.text(0.5, 0.04, 'data', ha='center', va='center')
fig.text(0.06, 0.5, 'target', ha='center', va='center', rotation='vertical')
fig.suptitle("Regression", fontsize=14)
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 10), sharey=True)
for ix, svr in enumerate(svrs):
    z  = svr.fit(X, y).predict(X)
    axes[ix].scatter(y,
                 z,
                 facecolor="none", edgecolor=model_color[ix], s=50,
                 label=kernel_label[ix]+":"+"{0:.2f}".format(metric(y,z))) 
    
    axes[ix].plot([np.min(y),np.max(y)],[np.min(y),np.max(y)])
    axes[ix].legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
                    ncol=1, fancybox=True, shadow=True)
    axes[ix].grid()

fig.text(0.5, 0.04, 'target', ha='center', va='center')
fig.text(0.06, 0.5, 'prediction', ha='center', va='center', rotation='vertical')
fig.suptitle("Regression", fontsize=14)
plt.show()