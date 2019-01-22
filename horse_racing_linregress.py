# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 23:43:31 2019

@author: u5326
"""
#単回帰分析
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from sklearn import linear_model
import statsmodels.api as sm
import statsmodels
from statsmodels.stats.outliers_influence import *
import pandas as pd

def linregress(x, y):
    result = scipy.stats.linregress(x, y)
    print('傾き=', result.slope.round(4), '切片=', result.intercept.round(4),\
          '信頼係数=', result.rvalue.round(4), 'p値=', result.pvalue.round(4),\
          '標準誤差=', result.stderr.round(4))
    
    plt.figure(figsize=(5,5))
    b = result.slope
    a = result.intercept
    plt.plot(x, [b * u + a for u in x])
    plt.scatter(x,y)
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

def linregress_sm(x, y):
    model = sm.OLS(y, sm.add_constant(x))
    results = model.fit()
    print(results.summary())
    print('p-values\n', results.pvalues)
    a, b = results.params
    print('a', a.round(4), 'b', b.round(4))

def regress(X,Y):
    model3 = sm.OLS(Y, sm.add_constant(X))
    result3 = model3.fit()
    print(result3.summary())
    print(result3.pvalues)
    print('=====================================================')
    num_cols = model3.exog.shape[1]#説明変数の列数
    vifs = [variance_inflation_factor(model3.exog, i) for i in range(0, num_cols)]
    pdv = pd.DataFrame(vifs, index=model3.exog_names, columns=['VIF'])
    print(pdv)