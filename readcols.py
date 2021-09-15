# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 17:04:18 2021

@author: gabri
"""


import pandas as pd 


df = pd.read_csv('./complete_df.csv')
finalDf = pd.read_csv('./finaldf.csv')

print(finalDf['issue_type'].unique())
# for i,col in enumerate(df.columns.tolist()):
#     print("index {} for column {}".format(i,col))
    
# print(df.info())