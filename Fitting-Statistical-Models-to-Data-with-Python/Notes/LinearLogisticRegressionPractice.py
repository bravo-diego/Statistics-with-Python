# Linear and Logistic Regression Modeling using NHANES Dataset

# Regression analysis aims to explain the variation of one variable (also known as outcome, response or dependent variable) in terms of one or more explanatory variables (also known as predictors or independent variables). 

# It's important to keep in mind that many regression analyses are only capable of describing associations and it may be misleading to interpret such relationships as representing causal effects. 

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

from statsmodels.graphics.regressionplots import plot_ccpr 
from statsmodels.graphics.regressionplots import add_lowess

path = '/home/aspphem/Desktop/Statistics-with-Python/nhanes_2015_2016.csv' # path file
df = pd.read_csv(path) # read csv file
print(df.head()) # data preview
print(df.keys()) # column names

unused_variables = ['BPXSY1', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'DMDEDUC2', 'BMXBMI', 'SMQ020'] # drop unused columns and drop rows with any missing values
df = df[unused_variables].dropna()

# Linear Regression

# We will predict a subject's systolic blood pressure (outcome or dependent variable) from other variables relating to that subject such as age, weight, etc. (explanatory or independent variables).

linear_regression_model = sm.OLS.from_formula('BPXSY1 ~ RIDAGEYR', data = df) # define a linear regression model using least squares method; BPXSY1 column contains the 1st recorded measurement of systolic blood pressure for a subject, and RIDAGEYR contains the subject's age in years
result = linear_regression_model.fit() # fit the model
print(result.summary()) # show least squares regression results

# The coefficient value for the age variable was about 0.48, this indicates that when comparing two people whose ages differ by one year, the older person will on average have 0.48 units higher the systolic blood pressure than the younger person. There is an association between the systolic blood pressure and age in this population.

# Note that the standard deviation of around 18.5 is large compared with the regression coefficient of 0.48. Thus, while there is a substantial tendency for blood pressure to increase with age, there is also a great deal of variation. We shouldn't be surprised if we find a 40 years old with greater blood pressure than a 60 years old. 

# The primary summary statistic for assesing the strength of a predictive relationship in a linear regression model is the R-squared. In this case, is shown to be 0.207, this means that approx 21% of the variation in the systolic blood pressure is explained by the age variable. 

	# Note: The pearson correlation coefficient, calculated below, measures the strength and direction of a linear relationship between two continuous variables. The closer r is to +- 1, the stronger the linear relationship. While R-squared measures the proportion of variance in the dependent variable explained by the independent variable(s) in a regression model.

corr = df[['BPXSY1', 'RIDAGEYR']].corr()
print(corr.BPXSY1.RIDAGEYR**2) # correlation coefficient (r)

# In the case of a simple linear regression model, where there is only one independent variable (or covariate) and one dependent variable, the R-squared value of the regression model is equal to the square of the Pearson correlation coefficient (r) between the independent variable and the dependent variable.

# Adding a 2nd Variable to the Linear Model

# As mention above, systolic blood pressure is expected to be related to gender as well as to age, so we add the variable gender to the model.

df['RIAGENDRx'] = df.RIAGENDR.replace({1: 'Male', 2: 'Female'}) # modify the categories for gender column

linear_regression_model = sm.OLS.from_formula('BPXSY1 ~ RIDAGEYR + RIAGENDRx', data = df) # define a linear regression model with both age and gender as explanatory variables
result = linear_regression_model.fit() # fit the model
print(result.summary()) # show least squares regression results

# Note that we got the same age coefficient that we found in the model fitted above. The model also shows that comparing a male and a female person of the same age, the male will on average have 3.23 units greater systolic blood pressure than the female.

# It's very important to emphasize that both age and gender coefficients are only meaningful when comparing two people of the same gender or when comparing people of the same age, respectively. Moreover, these effects are additive, meaning that if we compare, say, a 50 year old man to a 40 year old woman, the man's systolic blood pressure will on average be around 7.93 units higher (i.e. 3.23 + 10 * 0.47), with the first term in this sum being attributable to gender, and the second term being attributable to age.

# When adding variables to a multiple linear regression model, the R-squared value can never decrease. It can either remain unchanged, or it can increase. In this example the difference was minimal, in other words, the gender variable seems to have to play only a small role in explaining the variation in the systolic blood pressure. 

# When using a categorical variable as a predictor in a regression model, it's recoded into dummy variables (or indicator variables). A dummy variable for a single level, say a, of a variable x, is a variable that is equal to 1 when x = a and is equal to 0 when x is not equal to a. 

# Adding a 3rd Variable to the Linear Model

# We add the body mass index (BMI) variable to the model predicting systolic blood pressure. 

linear_regression_model = sm.OLS.from_formula('BPXSY1 ~ RIDAGEYR + BMXBMI + RIAGENDRx', data = df) # define a linear regression model with age, gender, and BMI as explanatory variables
result = linear_regression_model.fit() # fit the model
print(result.summary()) # show least squares regression results

# Given two subjects with the same gender and age, and whose BMI differs by 1 unit, the person with greater BMI will have on average 0.31 units greater systolic blood pressure.

# Partial Residual Plot

# Diagnostic tool used in regression analysis to visualize the effect of an independent variable on the dependent variable after controlling for the influence of other predictors. Also it can be used to identify nonlinearity in the relationship between an independent variable and the dependent variable.

ax = plt.axes()
plot_ccpr(result, 'RIDAGEYR', ax)
ax.lines[0].set_alpha(0.2)
ax.lines[1].set_color('indianred')
ax.grid(True)
plt.show() # partial residual plot fixing gender and BMI variables; the x-axis corresponds to the values of the independent variable (age), while the y-axis corresponds to the sum of the residuals (unexplained variability) plus the component age coefficient * age variable

# When gender and BMI variables are held fixed, the average blood pressures of an 80 and 18 years old differ by around 30 mm/Hg. We can see that the deviations from the mean are somewhat smaller at the low end of the range compared to the high end of the range.

ax = plt.axes()
plot_ccpr(result, 'BMXBMI', ax)
ax.lines[0].set_alpha(0.2)
ax.lines[1].set_color('orange')
ax.grid(True)
plt.show() # partial residual plot fixing gender and age variables

# It seems to be less information about the systolic blood pressure related to BMI variable.

# Added Variable Plot

# Diagnostic tool used in multiple linear regression to examine the unique contribution of a specific independent variable to explaining the variability in the dependent variable. 

# In general, it plots the unexplained variance (var1) of the dependent variable (excluding the effect of the variable of interest), against the unexplained variance (var2) of the variable of interest. This visualization helps determine whether the unexplained variance in var2 can account for or explain the unexplained variance in var1.

model = sm.GLM.from_formula('BPXSY1 ~ RIDAGEYR + BMXBMI + RIAGENDRx', data = df) # fitting a GLM with systolic blood pressure as the dependent variable and age, BMI, and gender as the independent variables
result = model.fit() # estimate model parameters
print(result.summary())

fig = result.plot_added_variable("RIDAGEYR")
ax = fig.get_axes()[0]
ax.lines[0].set_alpha(0.2)
add_lowess(ax)
ax.grid(True)
plt.show() # added variable plot for age variable; the x-axis corresponds to the residuals of regress the indepdent variable of interest (age variable) on the BMI and gender variables, while the y-axis corresponds to the residuals of regress the dependent variable on the BMI and gender variables (excluding age variable)

# The red line is an estimte of the relationship between age and blood pressure. Unlike the relationship in the model, it's not forced to be linear, and there is in fact a hint that the shape is slightly flatter for the first years of age. This would imply that systolic blood pressure increases slightly more slowly for younger people, then begins increasing faster for older ones, let's say 30 years old.

# Logistic Regression

# Regression models for binary outcomes.

df['Smoke'] = df.SMQ020.replace({2: 0, 7: np.nan, 9: np.nan}) # smoking and non-smoking are coded as 1 and 0, respectively, and responses like don't know and refused to answer are coded as missing values

# Logistic regression models provides a model for the odds of an event happening. Recall that if an event has probability p, then the odds for this event is p/(1 - p). The odds is a mathematical transformation of the probability onto a different scale.

smoking_proportions = pd.crosstab(df.RIAGENDRx, df.Smoke).apply(lambda x: x/x.sum(), axis = 1)
smoking_proportions["odds"] = smoking_proportions.loc[:, 1] / smoking_proportions.loc[:, 0] # add odds column of smoking for women and men
print(smoking_proportions) # probability that a woman has ever smoked is lower than the probability that a man has ever smoked (32% against 53%)

# It's common to work with odds ratios when comparing two groups, this is just the odds for one group divided by the odds for the other group.

print(smoking_proportions.odds.Male/smoking_proportions.odds.Female)

# It's conventional to work with odds on the logarithmic scale, by transforming the odds into log odds, we can express this relationship as a linear equation. 

smoking_proportions['logodds'] = np.log(smoking_proportions.odds)
print(smoking_proportions)

# Fitting a Logistic Regression Model

logistic_regression_model = sm.GLM.from_formula('Smoke ~ RIAGENDRx', family = sm.families.Binomial(), data = df) # fitting a GLM with smoking at least 100 cigarettes as the dependent variable and gender as the independent variable
result = logistic_regression_model .fit() # estimate model parameters
print(result.summary())

# Note that the logistic regression coefficient for male gender is exactly equal to the difference between the log odds statistics for males and females.  The model explicitly defines the regression coefficient​ as the change in log odds when moving from the reference group (females) to the group of interest (males). Thus, regression coefficient​ directly captures this difference.

print(smoking_proportions.logodds.Male - smoking_proportions.logodds.Female)

# Adding Additional Covariates

logistic_regression_model = sm.GLM.from_formula('Smoke ~ RIDAGEYR + RIAGENDRx', family = sm.families.Binomial(), data = df) # fitting a GLM with smoking at least 100 cigarettes as the dependent variable and both gender and age as the independent variables
result = logistic_regression_model.fit() # estimate model parameters
print(result.summary())

# The log odds for smoking increases by 0.017 for each year of age. Remember that this effect is additive, e.g. comparing two people whose ages differ by 20 years, the log odds of the older person smoking will be around 0.34 units greater than the log odds for the younger person smoking, adn the odds for the older person smoking will be around exp(0.34) = 1.4 times greater than the odds for the younger person smoking. 

# Note: Here the additivity is on the scale of log odds, not odds or probabilities. We can just exponentiate to convert these effects from the log odds scale to the odds scale. 

# In this case the regression coefficient changed just a bit, but in general, regression coefficients can change a lot when adding or removing other variables from a model. But recall that the coefficients for age and gender both have interpretations in terms of conditional log odds.

# Disclaimer: NHANES data are a "complex survey" and are accompanied with survey desing information such as weights, strata, and clusters. In many analyses, this survey design information should be used in the analysis to properly reflect the target population. To introduce how linear and logistic regression are used with independent data samples, or with convenience samples, we will not incorporate the survey structure of the NHANES sample into the analyses conducted here.

