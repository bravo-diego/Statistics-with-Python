# Basic techniques for exploring data using methods for understanding multivariate relationships

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("/home/aspphem/Desktop/StatisticsPythonCourse/Datasets/nhanes_2015_2016.csv")

# Quantitative Bivariate Data

# Bivariate data arise when every unit of analysis (i.e. a person in the NHANES dataset) is assessed with respect to two traits. 

sns.regplot(x = "BMXLEG", y = "BMXARML", data = df, fit_reg = True, scatter_kws = {"alpha": 0.2}) # a scatterplot is a very common and easily-understood visualization of quantitative bivariate data

plt.show()

sns.jointplot(x = "BMXLEG", y = "BMXARML", kind = 'kde', data = df) # plot of the density of points; plot margins show the densities for the arm lengths and leg lengths separately, while the plot in the center shows their density jointly

plt.show()

# The plot also shows the Pearson correlation coefficient between the arm length and leg length (0.62); Pearson correlation coefficient ranges from -1 to 1, a correlation of 0.62 would be considered a moderately strong positive dependence

sns.jointplot(x = "BPXSY1", y = "BPXDI1", kind = 'kde', data = df) # systolic and diastolic blood pressure are more weakly correlated (0.32); this weaker correlation indicates that some people have unusually high systolic blood pressure but have average diastolic blood pressure, and vice versa

plt.show()

# Most human characteristics are complex, they vary by gender, age, ethnicity and other factors (heterogeneity); when heterogeneity is present, it is usually productive to explore the data more deeply by stratifying on relevant factors

df["RIAGENDRx"] = df.RIAGENDR.replace({1: "Male", 2: "Female"})

sns.FacetGrid(df, col = "RIAGENDRx").map(plt.scatter, "BMXLEG", "BMXARML", alpha = 0.4).add_legend() # relationship between leg length and arm length; stratifying by gender

plt.show()

# Gender-stratified plot indicates that men tend to have somewhat longer arms and legs than women (this is reflected in the fact that the cloud of points on the left (men) is shifted slightly up and to the right relative to the cloud of points on the right (women))

print(df.loc[df.RIAGENDRx == "Female", ["BMXLEG", "BMXARML"]].dropna().corr()) # corr method of a data frame calculates the correlation coefficients for every pair of variables in the data frame; this returns a correlation matrix, which is a table containing the correlations between every pair of variables in the data set
print(df.loc[df.RIAGENDRx == "Male", ["BMXLEG", "BMXARML"]].dropna().corr())

# Consistent with the scatterplot, a slightly weaker correlation between arm length and leg length in women (compared to men) can be seen by calculating the correlation coefficient separately within each gender

# Stratifying data by both gender and ethnicity, this results in 10 total strata, since there are 2 gender strata and 5 ethnicity strata. 

sns.FacetGrid(df, col = "RIDRETH1", row = "RIAGENDRx").map(plt.scatter, "BMXLEG", "BMXARML", alpha = 0.5).add_legend()

plt.show()

# These scatterplots reveal differences in the means as well a differences in the degree of association (correlation) between different pairs of variables; some ethnic groups tend to have longer/shorter arms and legs than others, the relationship between arm length and leg length within genders is roughly similar across the ethnic groups

# Categorical Bivariate Data

df["DMDEDUC2x"] = df.DMDEDUC2.replace({1: "<9", 2: "9-11", 3: "HS/GED", 4: "Some college/AA", 5: "College", 7: "Refused", 9: "Don't know"}) # text labels instead of numbers to represent categories

df["DMDMARTLx"] = df.DMDMARTL.replace({1: "Married", 2: "Widowed", 3: "Divorced", 4: "Separated", 5: "Never Married", 6: "Living w/partner", 77: "Refused"})

db = df.loc[(df.DMDEDUC2x != "Don't know") & (df.DMDMARTLx != "Refused"), :]

x = pd.crosstab(db.DMDEDUC2x, df.DMDMARTLx)
print(x)

# The results will be easier to interpret if we normalize the data; a contingency table can be normalized in three ways, we can make the rows sum to 1, the columns sum to 1, or the whole table sum to 1

x.apply(lambda z: z/z.sum(), axis = 1) # normalize within rows
print(x) # proportion of people in each educational attainment category who fall into each group of the marital status variable

x.apply(lambda z: z/z.sum(), axis = 0) # normalize within columns
print(x) # proportion of people with each marital status group who have each level of educational attainment

# It's quite plausible that there are gender differences in the relationship between educational attainment and marital status

print(db.groupby(["RIAGENDRx", "DMDEDUC2x", "DMDMARTLx"]).size().unstack().fillna(0).apply(lambda x: x/x.sum(), axis = 1)) # 1) group data by every combination of gender, education and marital status (group by using respective name columns); 2) count the number of people in each cell using the size method; 3) pivot the marital status into the columns (unstack); 4) fill empty cells with 0; 5) normalize data by row

# This analysis yields some interesting trends, notably that women are much more likely to be widowed or divorced than men. Marital status is associated with many factors, including gender and educational status, but also varies strongly by age and birth cohort

# Mixed Categorical and Quantitative Data

# Another situation that commonly arises in data analysis is when we wish to analyze bivariate data consisting of one quantitative and one categorical variable. To illustrate this, consider the relationship between marital status and age in the NHANES data. Specifically, we consider the distribution of ages for people who are currently in each marital status category

plt.figure(figsize = (12, 4))
sns.boxplot(x = db.DMDMARTLx, y = db.RIDAGEYR) # specify x (values) and y (index) 
plt.show() # widowed people tend to be older, and never-married people tend to be younger

# When we have enough data, a 'violin plot' gives a bit more insight into the shapes of the distributions to a traditional boxplot

plt.figure(figsize = (12, 4))
sns.violinplot(x = db.DMDMARTLx, y = db.RIDAGEYR) # specify x (values) and y (index) 
plt.show()
