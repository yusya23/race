# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 11:29:42 2019

@author: kagitani
"""

import pandas as pd
import os
import datetime

#　pathの設定
os.chdir("C:\\Users\\kagitani\\Documents\\kaggle\\horse_race\\hong-kong-horse-racing")

horse_data = pd.read_csv('dataset/race_result_horse.csv', encoding='utf-8')
race_data = pd.read_csv('dataset/race_result_race.csv', encoding='utf-8')

all_data = pd.merge(horse_data, race_data, on='race_id', how='outer')
data = all_data.drop(['src', 'incident_report','running_position_5', 'running_position_6', 'running_position_4'], axis=1)
data = data.dropna()


data_K019 = data[data['horse_id']=="K019"]
def get_date(data, year=[], month=[], day=[]):
    for i in data["race_date"]:
        i = str(i)
        date = i.split('-')
        year.append(date[0])
        month.append(date[1])
        day.append(date[2])
    return year, month, day

year, month, day = get_date(data_K019)

    