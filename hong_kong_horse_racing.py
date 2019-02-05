# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:29:42 2019

@author: kagitani
"""

import pandas as pd
import os
import datetime
import numpy as np
from sklearn.model_selection import train_test_split

#　pathの設定
#os.chdir("C:\\Users\\kagitani\\Documents\\kaggle\\horse_race\\hong-kong-horse-racing")
def main():
    os.chdir("C:\\Users\\kagitani\\Documents\\race-master\\race-master")
    
    #生データ読み込み
    horse_data = pd.read_csv('dataset/race-result-horse.csv', encoding='utf-8')
    race_data = pd.read_csv('dataset/race-result-race.csv', encoding='utf-8')
    #データ連結
    all_data = pd.merge(horse_data, race_data, on='race_id', how='outer')
    #NANが多く含まれるcolumnを削除
    data = all_data.drop(['src', 'incident_report','running_position_5', 'running_position_6', 'running_position_4'], axis=1)
    #NANが含まれる行を削除
    data = data.dropna()
    
    #######前のレースからの経過日数のcolumn追加############
    #race_dateを年、月、日に分割
    def get_date(data, year=[], month=[], day=[]):
        for i in data:
            i = str(i)
            date = i.split('-')
            year.append(date[0])
            month.append(date[1])
            day.append(date[2])
        return year, month, day
    
    date = list(pd.Series(data['race_date']))
    year, month, day = get_date(date)
    data['year'] = pd.Series(year)
    data['month'] = pd.Series(month)
    data['day'] = pd.Series(day)
    
    #馬ごとに日付順に並び替え
    data = data.dropna()
    data_sort = data.sort_values(by=['horse_id', 'race_date'], ascending=True).reset_index()
    
    #経過日数の計算
    interval = []
    horse_id_prev = 'x'
    date_ymd_prev = 'x'
    for i in range(len(data_sort)):
        if data_sort['horse_id'][i] == horse_id_prev:
            date_ymd = datetime.date(int(data_sort['year'][i]), int(data_sort['month'][i]), int(data_sort['day'][i]))
            interval.append((date_ymd - date_ymd_prev).days)
    
        else:
            interval.append(0)
        
        horse_id_prev = data_sort['horse_id'][i]
        date_ymd_prev = datetime.date(int(data_sort['year'][i]), int(data_sort['month'][i]), int(data_sort['day'][i]))
    data_sort['interval'] = pd.Series(interval)
    ####################################################
    
    #race_classのダミー変数
    race_class = []
    for i in range(len(data_sort)):
        if 'Class 1' in data_sort['race_class'][i]:
            race_class.append(1)
        elif 'Class 2' in data_sort['race_class'][i]:
            race_class.append(2)
        elif 'Class 3' in data_sort['race_class'][i]:
            race_class.append(3)
        elif 'Class 4' in data_sort['race_class'][i]:
            race_class.append(4)
        elif 'Class 5' in data_sort['race_class'][i]:
            race_class.append(5)
        else:
            race_class.append(0)
    data_sort['race_class_dummy'] = pd.Series(race_class)
    
    #馬場状態のダミー変数
    track_condition = []
    for i in range(len(data_sort)):
        if data_sort['track_condition'][i] == 'FAST' or data_sort['track_condition'][i] == 'GOOD' or data_sort['track_condition'][i] == 'GOOD TO FIRM':
            track_condition.append(0)
        elif data_sort['track_condition'][i] == 'GOOD TO YIELDING' or data_sort['track_condition'][i] == 'YIELDING':
            track_condition.append(1)
        elif data_sort['track_condition'][i] == 'WET SLOW' or data_sort['track_condition'][i] == 'YIELDING TO SOFT':
            track_condition.append(2)
        else:
            track_condition.append(3)
    data_sort['track_condition_dummy'] = pd.Series(track_condition)
    
    #不要なcolumnを削除
    data_processed = data_sort.drop(['index', 'jockey', 'horse_name', 'horse_id', 'trainer', 'length_behind_winner', 'running_position_1', 'running_position_2', 'running_position_3', 'finish_time', 'race_id', 'race_date', 'race_course', 'race_class', 'track_condition', 'race_name', 'track', 'sectional_time', 'year', 'day'], axis=1)
    
    #ポンドからkg表記に変換
    kinryou = []
    horse_weight = []
    for i in range(len(data_processed)):
        kinryou.append(int(data_processed['actual_weight'][i]) // 2.205)
        horse_weight.append(int(data_processed['declared_horse_weight'][i]) // 2.205)
    data_processed['kinryou'] = pd.Series(kinryou)
    data_processed['horse_weight'] = pd.Series(horse_weight)
    data_processed = data_processed.drop(['actual_weight', 'declared_horse_weight'], axis=1)
    
    #数字以外を除外
    data_new = data_processed
    for i in range(len(data_processed)):
        if data_processed['finishing_position'][i].isdecimal() == False:
            data_new = data_new.drop(i, axis=0)
            #print(i)
    data_completed = data_new.reset_index()
    data_completed = data_completed.drop(['index'], axis=1)
    
    #test,train分割
    label = data_completed['finishing_position'].astype(np.float32)-1
    label_num = np.array(label)
    dataset = data_completed.drop(['finishing_position'], axis=1).astype(np.float32)
    dataset_num = np.array(dataset)
    
    #データセットのシャッフル
    rand = np.random.permutation(len(dataset_num))
    dataset_num = dataset_num[rand]
    label_num = label_num[rand]
    X_train, X_test, y_train, y_test = train_test_split(
            dataset_num, label_num,
            test_size=0.4, random_state=0)
    return data_completed, label, dataset, X_train, X_test, y_train, y_test