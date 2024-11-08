# Regression Methods for Data Modeling

# By fitting models to data we are able to

	# Estimate distributional properties of variables, potentially conditional on other variables.

	# Summarize relationships between variables, and make inferential statements about those relationships.
	
	# Predict values of variables of interest conditional on values of other predictor variables, and characterize prediction uncertainty. 

# Most methods of Regression Analysis aim to explain the variation in a specific variable (dependent variable/outcome/response) in terms of other variables (independent variables/regressors/predictors/explanatory variables).  

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes # loading libraries

data = load_diabetes() # load diabetes dataset
df = pd.DataFrame(data = data.data, columns = data.feature_names) # create a pandas dataframe

# Exploring the Dataset Structure

print(df.shape) # dataframe dimensions; it has 442 rows and 10 columns
print(df.keys()) # column names
print(df.head(2)) # dataframe preview

	# Check for Missing Values

print(df.isnull().sum()) # check for null values; there isn't missing data

# Most applications of regression analysis involve multiple explanatory variables. The explanatory variables may be associated with each other, as well as being associated with the outcome. Thus it's of interest to consider the marginal association of each independent variable with the response

correlation_matrix = df.corr().round(2)
correlation_matrix *= 1 + np.diag(np.nan*np.ones(10))
sns.heatmap(data = correlation_matrix, annot = True, cmap = 'cividis')
plt.show() # correlation values close to 1 signify a strong positive relationship; correlation values close to -1 indicates a strong negative relationship

# We can see that there are substantial correlations between the dependent variable s6 (glu/blood sugar levels) and each of the independent variables. It's also clear that the explanatory variables are correlated with each other.

# Another way to see the relationship between two variables is viewing a scatterplot. For large datasets, scatterplots are not very informative due to overplotting. We can address overplotting by visualizing the joint distribution between the explanatory variable and the response variable using two-dimensional density estimates.

for column in df.columns: # iterate over the columns of the dataframe
    if column in ["s6", "sex"]: # exclude s6 and sex columns
        continue
    plt.figure()
    plt.grid(True)
    sns.kdeplot(x = column, y = "s6", data = df) # plot a kernel density estimate (KDE) plot
    plt.show()

# Fitting a Linear Model using Ordinary Least Squares Method

# A linear model expresses the conditional mean of the dependent variable as a linear function of the independent variables. It does this by estimating coefficients or slopes for each independent variable, along with one coefficient known as intercept. 

linear_regression_model = sm.OLS.from_formula("s6 ~ age + sex + bmi + bp", df)
fitted_model = linear_regression_model.fit()
print(fitted_model.summary()) # OLS regression model results

# We can interpret each coefficient as the expected change of the dependent variable for each unit change in an independent variable, while holding the other independent variables fixed. 

# In this case we obtained a coefficient value for age of 0.1654. This indicates that if we compare two people of the same sex, BMI, and blood pressure, and one of these people is one year older than the other, then the blood glucose of the older person will on average be 0.1654 units greater than the blood glucose of the younger person.

# Note: A key aspect of multiple regression analysis is that the association attributed to each explanatory variable corresponds to a setting where the other explanatory variables are held fixed (i.e. ignoring the roles of other variables).

