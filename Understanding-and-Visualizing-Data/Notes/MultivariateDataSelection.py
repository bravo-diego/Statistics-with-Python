# How to select data frame subsets from multivariate data

import numpy as np
import pandas as pd

df = pd.read_csv("/home/aspphem/Desktop/StatisticsPythonCourse/Datasets/nhanes_2015_2016.csv")

print(df.head())
print(df.columns)

cols = ['BMXWT', 'BMXHT', 'BMXBMI', 'BMXLEG', 'BMXARML', 'BMXARMC', 'BMXWAIST']

updated_df = df[cols] # use [] notation to keep columns
print(updated_df.head())

print(df.loc[:, cols].head())

BMXWAIST_median = pd.Series.median(updated_df['BMXWAIST'])
print(BMXWAIST_median)

# Look at rows who 'BMXWAIST' is larger than the median

print(updated_df[updated_df['BMXWAIST'] > BMXWAIST_median].head())

# Adding another condition; 'BMXLEG' must be less than 32

fst_condition = updated_df['BMXWAIST'] > BMXWAIST_median
snd_condition = updated_df['BMXLEG'] < 32

print(updated_df[fst_condition & snd_condition].head()) # using [] method
print(updated_df.loc[fst_condition & snd_condition, :].head()) # using loc method

tmp = updated_df[fst_condition & snd_condition].head()
tmp.index = ['a', 'b', 'c', 'd', 'e']
print(tmp)

# Difference between loc and iloc

print(tmp.loc[['a', 'b'], 'BMXLEG'])
print(tmp.iloc[[0, 1], 3])
