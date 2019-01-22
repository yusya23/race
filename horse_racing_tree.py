# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 22:14:35 2019

@author: u5326
"""

# モデル設定と訓練データを使った学習
from sklearn import tree
# 訓練データでの正解率（学習検証）
from sklearn import metrics

def horse_racing_tree(X_train, y_train, max_depth):
    clf = tree.DecisionTreeClassifier(max_depth=max_depth) # 決定木モデル（最大深さ3）
    clf = clf.fit(X_train, y_train) # 訓練データで学習
    predict_train = clf.predict(X_train)
    ac_score = metrics.accuracy_score(y_train, predict_train)
    print('train score: {0:.2f}%'.format(ac_score * 100))
    return ac_score