# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 23:52:19 2019

@author: u5326
"""

import pandas as pd
import numpy as np
import scipy.stats as st
#from rpy2.robjects import pandas2ri
#pandas2ri.activate()
#from rpy2.robjects import r
#
#t = r['Titanic']
#クロス集計表相関係数０，１(ファイ係数)
def corr(apple, orange):
    apple = [0, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    orange = [1, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    print('相関係数', np.corrcoef(apple, orange)[0, 1].round(4))
#numpy配列からケンドールτ
def kendall(x, y):
    tau, p_value = st.kendalltau(x, y)
    print('tau', tau.round(4), 'p_value', p_value)

#クロス集計表からケンドールτ
#DataFrameから集計表作成
def kendall2(pd_cross):
    #pd_cross = np.array(pd.crosstab(line, column))
    z = [[[i, j]] * (pd_cross[i, j]) for i in range(pd_cross.shape[0]) for j in range(pd_cross.shape[1])]
    #zを二重リストから平らなリストに変更
    tdata = []
    for v in z:
        tdata.extend(v)
    x = [u[0] for u in tdata]
    y = [u[1] for u in tdata]
    tau, p_value = st.kendalltau(x, y)
    print('tau', tau.round(4), 'p_value', p_value) 