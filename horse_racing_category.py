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

def kendall(x, y):
    tau, p_value = st.kendalltau(x, y)
    print('tau', tau.round(4), 'p_value', p_value)
    