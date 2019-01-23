# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 10:31:02 2019

@author: kagitani
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import scale

def PCA_(dataset):
    n_components = dataset.shape[1]
    pca = PCA(n_components = n_components)
    pca.fit(scale(dataset))
    
    print('主成分', pca.components_.round(4))
    print('平均', pca.mean_.round(4))
    print('分散', pca.explained_variance_.round(4))
    print('共分散', pca.get_covariance().round(4))
    print('寄与率', pca.explained_variance_ratio_.round(4))
    print('累積寄与率', np.cumsum(pca.explained_variance_ratio_).round(4))
    print('標準偏差\n', pd.DataFrame([math.sqrt(u) for u in pca.explained_variance_]).T.round(4))
    return pca.components_, pca.explained_variance_, pca.get_covariance(), pca.explained_variance_ratio_, pca


def PCA_graph(dataset, data0, data1, data2):
    n_components = dataset.shape[1]
    pca = PCA(n_components = n_components)
    pca.fit(dataset)
    transformed0 = pca.transform(data0)
    transformed1 = pca.transform(data1)
    transformed2 = pca.transform(data2)
    color = ['red', 'blue', 'green']
    marker = ['x', '.', '+']
    plt.scatter([u[0] for u in transformed0], [u[1] for u in transformed0], c=color[0], label='0', marker=marker[0])
    plt.scatter([u[0] for u in transformed1], [u[1] for u in transformed1], c=color[1], label='1', marker=marker[1])
    plt.scatter([u[0] for u in transformed2], [u[1] for u in transformed2], c=color[2], label='2', marker=marker[2])
    plt.title('データの主成分分析')
    plt.xlabel('pc1')
    plt.ylabel('pc2')
    plt.legend()
    plt.show()

def biplot(score, coeff, pcax, pcay, labels=None):
    pca1 = pcax - 1
    pca2 = pcay - 1
    xs = score[:, pca1]
    ys = score[:, pca2]
    n = score.shape[1]
    scalex = 2.0 / (xs.max() - ys.min())
    scaley = 2.0 / (ys.max() - ys.min())
    for i in range(len(xs)):
        plt.text(xs[i] * scalex, ys[i] * scaley, str(i+1), color='k', ha='center', va='center')
    for i in range(n):
        plt.arrow(0, 0, coeff[i, pca1], coeff[i, pca2], color='r', alpha=1.0)
    if labels is None:
        plt.text(coeff[i, pca1] * 1.10, coeff[i, pca2] * 1.10, 'Var' + str(i+1), color='k', ha='center', va='center')
    else:
        plt.text(coeff[i, pca1] * 1.10, coeff[i, pca2] * 1.10, labels[i], color='k', ha='center', va='center')
    plt.xlim(min(coeff[:, pca1].min() - 0.1, -1.1), max(coeff[:, pca1].max() + 0.1, 1.1))
    plt.ylim(min(coeff[:, pca2].min() - 0.1, -1.1), max(coeff[:, pca2].max() + 0.1, 1.1))
    plt.xlabel('PC{}'.format(pcax))
    plt.ylabel('PC{}'.format(pcay))
    plt.grid()
    plt.show
    
