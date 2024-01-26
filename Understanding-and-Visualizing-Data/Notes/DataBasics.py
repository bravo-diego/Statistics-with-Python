import numpy as np
import pandas as pd

# Data Management 

df = pd.read_csv("/home/aspphem/Desktop/StatisticsPythonCourse/Datasets/Cartwheeldata.csv") # importing data via pandas library

print(type(df)) # object type

print(df.head(5)) # viewing data

print(df.columns)

    # DataFrame.loc 

# .loc attribute is used to access a group of rows and columns by label(s); [row(s), column(s)]

print(df.loc[:,"CWDistance"]) # return all observations (rows) of columns 'CWDistance'

print(df.loc[:, ["CWDistance", "Height", "Wingspan"]]) # return all observations of multiple columns

print(df.loc[:5, ["CWDistance", "Height", "Wingspan"]]) # return the first 6 observations

print(df.loc[10:15]) # return a range of rows for all columns

    # DataFrame.iloc
    
# .iloc attribute is used to access a group of rows and columns by position

print(df.iloc[:4]) # return the first 4 observations

print(df.iloc[1:5, 2:4]) # return rows range 1 to 4 of columns 2 ('Gender') and 3 ('GenderGroup') 

print(df.dtypes) # data types of our data frame columns

print(df.Gender.unique()) # list unique values in the 'Gender' column

print(df.loc[:, ["Gender", "GenderGroup"]])

print(df.groupby(["Gender", "GenderGroup"]).size()) # these two fields essentially portray the same information

# To confirm that we have actually obtained the data that we are expecting, we can display the shape of the data frame

print(df.shape) # rows, columns

# The extract all the values for one variable, the following approaches are equivalent

x = df["Gender"]
y = df.loc[:, "Gender"]
z = df.Gender
w = df.iloc[:, 2] # Gender is in column 2

print(df["CWDistance"].max())
print(df.loc[:, "CWDistance"].max())
print(df.CWDistance.max())
print(df.iloc[:, 8].max())

    # Missing Values
    
# Pandas has functions that can be used to identify where the missing and non-missing values are located in a data frame.

print(pd.isnull(df.CWDistance).sum())
print(pd.notnull(df.CWDistance).sum())
