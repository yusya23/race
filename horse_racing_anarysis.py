# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:04:51 2019

@author: kagitani
"""

import horse_racing_NN as hrnn
import horse_racing_linregress as hrl
import horse_racing_category as hrc
import horse_racing_tree as hrt
import hong_kong_horse_racing as hkhr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA

data_completed, label, dataset, X_train, X_test, y_train, y_test = hkhr.main()
#決定木
hrt.horse_racing_tree(X_train, y_train, 3)
#NN
hrnn.horse_racing_NN(X_train, X_test, y_train, y_test)
#NN2
hrnn.horse_racing_NN2(X_train, X_test, y_train, y_test)
#linregress
X = np.array(data_completed['win_odds'].astype(np.float16))
Y = np.array(data_completed['finishing_position'].astype(np.int8))
hrl.linregress(X, Y)
hrl.linregress_sm(X, Y)

x = dataset
y = label
hrl.regress(x, y)
#category
line = np.array(data_completed['finishing_position'].astype(np.int8))
column = np.array(data_completed['draw'].astype(np.int8))
#numpy配列からケンドールτ
hrc.kendall(line, column)
#クロス集計表からケンドールτ
pd_cross = np.array(pd.crosstab(line, column))
hrc.kendall2(pd_cross)

#主成分分析
import horse_racing_PCA as hrp

data_1 = data_completed[data_completed['finishing_position']=='1'].drop(['finishing_position'], axis=1)
data_7 = data_completed[data_completed['finishing_position']=='7'].drop(['finishing_position'], axis=1)
data_14 = data_completed[data_completed['finishing_position']=='14'].drop(['finishing_position'], axis=1)

finishing_position = ['1', '7', '14']
xlabel = 'interval'
ylabel = 'finishing_position'
plt.scatter(data_1[xlabel], data_1[ylabel], c='red', label='1', marker='x')
plt.scatter(data_7[xlabel], data_7[ylabel], c='blue', label='7', marker='.')
plt.scatter(data_14[xlabel], data_14[ylabel], c='green', label='14', marker='+')
plt.title('horse散布図')
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.legend()
plt.show()
#主成分表示
pca_components_, pca_explained_variance_, pca_get_covariance, pca_explained_variance_ratio_, pca = hrp.PCA_(dataset)
#次元削減後描写
hrp.PCA_graph(dataset, data_1, data_7, data_14)
#倍プロット描写
u = pd.DataFrame([[math.sqrt(u) for u in pca_explained_variance_]] * 9)
u0 = u[0][0]
pca_components = pd.DataFrame(pca_components_)
x = pca_components_[0, :] * u0
y = pca_components_[1, :] * u0
fuka = (np.array([x, y])).T
hrp.biplot(pca.transform(data_1), fuka, 1, 2, labels=data_1.columns)