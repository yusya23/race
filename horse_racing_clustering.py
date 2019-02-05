# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:28:02 2019

@author: kagitani
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt

pd.set_option('display.max_rows', 200)

X = np.array(X_test_df)
Z = linkage(X,'ward')
r = fcluster(Z, t=3, criterion='maxclust') # t；階層的クラスタリングで3種類に分類
import collections
print(collections.Counter(r))

# クラスタ番号はクラスタリングで適当につけられるので、元データの種類とは異なる
# 