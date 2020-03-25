#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 10 14:46:14 2020

@author: pawelwicherek
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pyod
import os
import seaborn as sns
os.chdir("/home/pawelwicherek/XAI/")

df = pd.read_csv("kc_house_data.csv")

r = pd.to_datetime(r)
df["czas"] = df.date.str[0:8].astype(int)
df = df.drop("id",axis=1)
df.price = df.price.astype(int)
df.zipcode = df.zipcode.astype("category")
y = df.price
df = df.drop(["price","date"],axis=1)
df.info()


sns.distplot(df.zipcode,kde=False)



########## podział na próby Test / Ucz
from sklearn.model_selection import train_test_split
seed = 484
test_size = 0.2
x_train, x_test, y_train, y_test \
    = train_test_split(data, y, test_size=test_size,
                       random_state=seed)


import sklearn.ensemble
lasy = sklearn.ensemble.RandomForestRegressor()


lasy.fit(x_train, y_train)
lasy.score(x_train,y_train)
lasy.score(x_test,y_test)

lasy_cv = sklearn.model_selection.cross_val_score(lasy, df, y, cv = 10)
lasy_cv.mean()

from sklearn.ensemble import AdaBoostRegressor as ADA

ada = ADA()
ada.fit(x_train,y_train)
ada.score(x_train,y_train)
ada.score(x_test,y_test)

#zachowuje się fatalnie, rezygnujemy


from pyearth import Earth
