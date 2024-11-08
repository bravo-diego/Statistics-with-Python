# Introduction to Multilevel and Marginal Regression

# This analysis is based on data from a longitudinal study of children with autism. We will consider how various factors interact with the socialization of a child with autism as they progress through the early stages of their life.

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm 
import matplotlib.pyplot as plt # loading libraries

path = '/home/aspphem/Desktop/Statistics-with-Python/autism.csv' # path file
df = pd.read_csv(path) # read csv file
df = df.dropna() # drop missing values
print(df.head(10)) # data preview
print(df.keys()) # column names

# The level of socialization, vsae column, tends to increase with age. To visualize this we can use a scatterplot.

sns.scatterplot(x = 'age', y = 'vsae', data = df)
plt.show()

# Since age takes on only a few distinct values, we can also visualize it using a series of boxplots.

sns.boxplot(x = 'age', y = 'vsae', data = df)
plt.show()

# Multilevel Linear Regression Models

# Fit a basic multilevel linear model, with vsae as the response variable and age and sicdegp as explanatory variables. We use C() sintax to force it to be treated as a categorical variable.

# This is a repeated measures longitudinal dataset, so we treat each child as a group. In this basic model we include only a random intercept to capture the within-child dependence.

multilevel_linear_model = sm.MixedLM.from_formula(
	formula = 'vsae ~ age + C(sicdegp)',
	groups = 'childid',
	data = df) # define a multilevel linear regression model

results = multilevel_linear_model.fit(reml = False) # estimate model parameters
print(results.summary())

# The coefficient for age is approx. 4.5, and is statistically different from zero. Thus, for a fixed level of sicdegp (the only other explanatory variable), the expected value of vsae increases by approximately 4.5 units with each additional year of age. Here sicdegp = 1 is the reference level, so, since the coefficient for sicdegp = 2 is not statistically different from zero, there is no evidence that the mean value of vsae differs between sicdegp = 1 and sicdegp = 2, when age is fixed (considering the reference level). However the estimate for sicdegp = 3 is strongly statistically different from zero, with a point estimate of approximately 21.4; i.e. children with sicdegp = 3 have approximately 21.4 points greater vsae than children with sicegp = 1 or sicdegp = 2, for a fixed age. 

# The model fitted above is just a random intercept model. Another possibility is that children also differ by the rate of change, or slope of vsae with respect to age. To capture this possibility, we include a random slope for age as follows.

df['age_cen'] = df['age'] - df['age'].mean() # when modeling random slopes for quantitative variables in a multilevel model, it is common to center the covariate

multilevel_linear_model_updated = sm.MixedLM.from_formula(
    formula = 'vsae ~ age + C(sicdegp)', 
    groups = 'childid', 
    re_formula = '1',
    vc_formula = {'age_cen': '0+age_cen'}, # variance component; the outcome vsae could vary across different childs, and this variability is associated with age
    data = df
) # define a multilevel linear regression model

updated_results = multilevel_linear_model_updated.fit(reml = False) # estimate model parameters
print(updated_results.summary())

# The variance structure now describes the unique child effect in terms of both a random intercept and a random age slope.

# Comparing Models in terms of Akaike Information Criterion (AIC)

print(results.aic)
print(updated_results.aic) # a lower AIC corresponds to a better fit; the model with the variance component for age variable has a substantially better fit

# Marginal Linear Regression Models

# Recall that in an analysis using GEE, we specify a working correlation structure. This represents an attempt to specify how the observations within a group are correlated.

marginal_linear_model = sm.GEE.from_formula(
    formula = 'vsae ~ age + C(sicdegp)',
    groups = 'childid',
    cov_struct=sm.cov_struct.Exchangeable(), # exchangeable correlation - any two observations on the same child have the same correlation between them
    data = df
    ).fit() # define a marginal linear regression model
print(marginal_linear_model.summary())

