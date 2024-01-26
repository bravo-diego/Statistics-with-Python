# Using Python and Pandas to perform some basic analyses with univariate data (2015-2016 wave of the NHANES study).

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

df = pd.read_csv("/home/aspphem/Desktop/StatisticsPythonCourse/Datasets/nhanes_2015_2016.csv") # importing data via pandas library

print(df.columns)

print(df.DMDEDUC2.value_counts()) # value counts method can be used to determine the number of times that each distinct value of a variable occurs in a data set; (i.e. frequency distribution). N O T E: value counts method excludes missing values

print(df.DMDEDUC2.value_counts().sum())
print(df.shape)

print(pd.isnull(df.DMDEDUC2).sum()) # count all the null (missing) values in the data set

# In some cases it is useful to replace integer codes with a text label that reflects the code's meaning. 

df["DMDEDUC2x"] = df.DMDEDUC2.replace({1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College", 7: "Refused", 9: "Don't know"})

print(df.DMDEDUC2x.value_counts())

df["RIAGENDRx"] = df.RIAGENDR.replace({1: "Male", 2: "Female"}) # relabel the gender variable

x = df.DMDEDUC2x.value_counts()
print(x/x.sum()) # proportions are more relevant than the number of people in each category

df["DMDEDUC2x"] = df.DMDEDUC2x.fillna("Missing") # create 'Missing' category and assign all missing values to it  
x = df.DMDEDUC2x.value_counts()
print(x/x.sum())

df.BMXWT.dropna().describe() # numerical summary for a quantitative variable using the describe method 

# It's also possible to calculate individual summary statistics from one column of a data set using both pandas methods and numpy funtions

x = df.BMXWT.dropna() 
print(x.mean()) # pandas method
print(np.mean(x)) # numpy function

# Calculate the proportion of the NHANES sample who would be considered to have pre-hypertension (i.e. blood pressure between 120 and 139)

print(np.mean((df.BPXSY1 >= 120) & (df.BPXSY2 <= 139)))

# Graphical Summaries

sns.histplot(df.BMXWT.dropna())
plt.show()

sns.histplot(df.BPXSY1.dropna())
plt.show()

sns.boxplot(data = df.loc[:, ["BPXSY1", "BPXSY2", "BPXDI1", "BPXDI2"]]).set(ylabel= "Blood pressure in mm/Hg") # compare several distributions using a side by side boxplots
plt.show()

# One of the most effective ways to get more information out of a data set is to divide it into smaller, more uniform subsets, and analyze each of these strata on its own. We can then formally or informally compare the findings in the different strata. When working with human subjects, it is very common to stratify on demographic factors such as age, sex, and race

df["agerp"] = pd.cut(df.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80]) # create age strata based on these cut points
plt.figure(figsize=(12, 5))
sns.boxplot(x = "agerp", y = "BPXSY1", data = df)
plt.show()

# Stratifying on two factors (age and gender); group first by age and within the bands by gender

df["agerp"] = pd.cut(df.RIDAGEYR, [18, 30, 40, 50, 60, 70, 80])
plt.figure(figsize=(12, 5))
sns.boxplot(x = "RIAGENDRx", y = "BPXSY1", hue = "agerp", data = df)
plt.show()

# Stratification can also be useful when working with categorical variables

print(df.groupby("agerp")["DMDEDUC2x"].value_counts())

# We can also stratify jointly by age and gender.

dx = df.loc[~df.DMDEDUC2x.isin(["Don't know", "Missing"]), :] # eliminate rare/missing values
dx = dx.groupby(["agerp", "RIAGENDRx"])["DMDEDUC2x"]
dx = dx.value_counts()
dx = dx.unstack() # restructure the results from long to wide
dx = dx.apply(lambda x: x/x.sum(), axis = 1) # normalize within each stratum to get proportions
print(dx.to_string(float_format="%.3f")) # display only 3 decimal places
