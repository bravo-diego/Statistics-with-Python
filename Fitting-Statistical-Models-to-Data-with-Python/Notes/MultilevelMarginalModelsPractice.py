# Multilevel and Marginal Regression Models using NHANES Dataset

# Data can be dependent, or have a multilevel structure for a number of different reasons. We'll consider the NHANES dataset from the perspective of dependence, focusing in particular on dependence in the data that arises due to clustering.

# First we need to understand how NHANES data was collected. The data values were collected as a cluster sample. It means that the population of interest was first partitioned into groups, then a limited no. of these groups were selected, and finally a limited no. of individuals were selected from each of the sampled groups. 

# Since NHANES involves physical examinations, it's not practical to select a random sample from the entire US population, as this would involve conducting the examinations at thousands of dispersed locations. By utilizing cluster sampling, the NHANES staff can set up an examination center in each selected community, and assess many subjects at each center.

# Cluster sampling is NOT the only reason that dependence may exist between observations in a dataset. Many studied are longitudinal, meaning that each subject is assessed on multiple times. In this setting, we would expect these repeated measurements to be correlated. 

# We'll focus on multilevel modeling since NHANES data is a cluster sample. Recall that in any cluster sample, it is likely that observations within the same cluster are more similar than observations in different clusters.

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt # loading libraries

path = '/home/aspphem/Desktop/Statistics-with-Python/nhanes_2015_2016.csv' # path file
df = pd.read_csv(path) # read csv file
print(df.keys()) # column names
print(df.shape) # no. of rows; no. of columns

variables = ["BPXSY1", "RIDAGEYR", "RIAGENDR", "RIDRETH1", 
	     "DMDEDUC2", "BMXBMI", "SMQ020", "SDMVSTRA", "SDMVPSU"]
df = df[variables].dropna() # drop unused columns and missing values
print(df.shape) # no. of rows; no. of columns

# Note: For privacy reasons the county information information is not released with the data, instead we'll use masked variance units (MVUs), which are formed by combining subregions of different counties into artificial groups that aren't geographically contiguous. While MVUs are not the actual clusters of the survey, and are not truly contiguous geographic regions, they are selected to mimic these things.

df["group"] = 10*df.SDMVSTRA + df.SDMVPSU # MVUs identifiers can be obtained combining SDMVSTRA and SDMVPSU columns

# Marginal Models

	# Intraclass Correlation

# Observations within a cluster can be measured using a statistic called intraclass correlation coefficient (ICC). The ICC takes on values from 0 to 1, with 1 corresponding to perfect clustering (i.e. the values within a cluster are identical), and 0 corresponding to independence (i.e. the mean value within each cluster is identical across all the clusters).

# We can assess ICC using two regression techniques, marginal regression, and multilevel regression.

marginal_model = sm.GEE.from_formula('BPXSY1 ~ 1', groups = 'group', cov_struct = sm.cov_struct.Exchangeable(), data = df) # define a marginal regression model with exchangeable as variance-covariance structure
result = marginal_model.fit() # estimate model parameters
print(result.cov_struct.summary())

# While 0.03 would generally be considered to be very small as a Pearson correlation coefficient, it's not the case as an ICC. To illustrate that the ICC value of 0.03 are consistent with a presence of dependence, we simulate 10 sets of random data and calculate the ICC value for each set.

for i in range(10): # simulate 10 sets of random noise ~ N(0, 1) 
	df['noise'] = np.random.normal(size = df.shape[0])
	test_model = sm.GEE.from_formula('noise ~ 1', groups = 'group', cov_struct = sm.cov_struct.Exchangeable(), data = df)
	result = test_model.fit()
	print(result.cov_struct.summary())

# Estimated ICC for simulated random noise is concentrated near 0 (varying from -0.002 to 0.002). While the ICC values for the NHANES data are small, they are much larger than what we would expect to obtain if the observations were independent.

	# Conditional Intraclass Correlation
	
# We know that older people have higher systolic blood pressure than younger people. Also, some clusters may contain a slightly older or younger set of people than others. Thus, by controlling for age, we might anticipate that the ICC will become smaller.

marginal_model = sm.GEE.from_formula('BPXSY1 ~ RIDAGEYR', groups = 'group', cov_struct = sm.cov_struct.Exchangeable(), data = df) # adding age as a covariate in the model
result = marginal_model.fit()
print(result.cov_struct.summary()) # the ICC drops from 0.03 to 0.02

# By adding age as a covariate in the model, we're telling the model that some of the variability in the outcome is explained by age differences among individuals. In other words, the clusters tend to look more similar, because we've explained away a portion of the variability that was due to age differences rather than clustering.

# ICC drops even further when we add additional covariates that we know to be predictive of systolic blood pressure, i.e. much of the variability is due to individual characteristics rather than the clusters themselves.
	
# While the mean structure (i.e. the regression coefficients) can be estimated without considering the dependence structure of the data, the standard errors and other statistics relating to uncertainty will be wrong if we ignore dependence in the data.

# To illustrate this, we fit two models. The first one is a regression model with the ordinary least squares method, while the second one is a marginal regression model using the GEE approach which allows us to account for the dependence in the data.

df["RIAGENDRx"] = df.RIAGENDR.replace({1: 'Male', 2: 'Female'}) # recode RIAGENDR column

ols_regression_model = sm.OLS.from_formula('BPXSY1 ~ RIDAGEYR + RIAGENDRx + BMXBMI + C(RIDRETH1)', data = df) # define a multiple linear regression model using OLS method
ols_result = ols_regression_model.fit() # estimate model parameters

gee_regression_model = sm.GEE.from_formula('BPXSY1 ~ RIDAGEYR + RIAGENDRx + BMXBMI + C(RIDRETH1)', groups = 'group', cov_struct = sm.cov_struct.Exchangeable(), data = df) # define a marginal regression model using GEE approach
gee_result = gee_regression_model.fit() # estimate model parameters

estimates = pd.DataFrame({'OLS_params': ols_result.params, 'OLS_SE': ols_result.bse,
				  'GEE_params': gee_result.params, 'GEE_SE': gee_result.bse})
estimates = estimates[['OLS_params', 'OLS_SE', 'GEE_params', 'GEE_SE']]
print(estimates)

# As we mention before, point estimates are similar between the OLS and GEE fits of the model, but the standard errors tend to be larger in the GEE fit (the OLS parameter estimates remain in meaningful, but the standard errors don't). GEE estimates and standard errors are meaningful in the presence of dependence, as long as the dependence is exclusively between observations within the same cluster.

	# Marginal Logistic Regression Model
	
# GEE approach can also be used to fit any GLM in the presence of dependence.

df["smq"] = df.SMQ020.replace({2: 0, 7: np.nan, 9: np.nan}) # recode smoking to a binary variable

df['DMDEDUC2x'] = df.DMDEDUC2.replace({1: 'lt9', 2: 'x9_11', 3: 'HS', 4: 'SomeCollege', 5: 'College', 7: np.nan, 9: np.nan}) # recode DMDEDUC2 variable

glm_model = sm.GLM.from_formula('smq ~ RIDAGEYR + RIAGENDRx + C(DMDEDUC2x)', family = sm.families.Binomial(), data = df) # define a logistic regression model for comparison
glm_result = glm_model.fit() # estimate model parameters

gee_model = sm.GEE.from_formula('smq ~ RIDAGEYR + RIAGENDRx + C(DMDEDUC2x)', groups = 'group', family = sm.families.Binomial(), cov_struct = sm.cov_struct.Exchangeable(), data = df) # define a marginal model using GEE approach
gee_result = gee_model.fit(start_params = glm_result.params) # estimate model parameters

estimates = pd.DataFrame({'OLS_params': glm_result.params, 'OLS_SE': glm_result.bse, 'GEE_params': gee_result.params, 'GEE_SE': gee_result.bse})
estimates = estimates[['OLS_params', 'OLS_SE', 'GEE_params', 'GEE_SE']]
print(estimates)

# GLM and the GEE give very similar estimates for the regression parameters (i.e.  the main trend in the data is captured similarly by both methods). However the standard errors obtained using GEE are somewhat larger than those obtained using GLM. GLM model leads to an underestimation of the standard errors, making the model appear more precise than it actually is.

# In summary, GEE approach: 

	# Gives us insight into the dependence structure of the data.
	
	# Uses the dependence structure to obtain meaningful standard errors of the estimated model parameters.
	
	# Uses the dependence structure to estimate the model parameters more accurately.
	
# Note: Is clear that GEE approach should in general have an efficiency advantage over GLM model, but GLM model estimates remain valid and cannot be completely dismissed.

# Multilevel Models

# A multilevel model is usually expressed in terms of random effects. These are variables that we don't observe, but that we can nevertheless incorporate into a statistical model. In this context we can imagine that each cluster has a random effect that is shared by all observations in that cluster.

	# Random Intercept Model

multilevel_model = sm.MixedLM.from_formula('BPXSY1 ~ RIDAGEYR + RIAGENDRx + BMXBMI + C(RIDRETH1)', groups = 'group', data = df) # define a multilevel regression model; the groups argument specifies that the model will include a random intercept for each unique value in the group variable
result = multilevel_model.fit() # estimate model parameters
print(result.summary())

# The variance structure parameters are what distinguish a mixed model from a marginal model. The variance for groups, estimated to be 3.615, means that if we were to choose two groups at random, their random effects would differ on average by around 2.69 (Var(X1 - X2) = Var(X1) + Var(X2) = 2\sigma^2; thus \sqrt{2*3.6115}). 

print(result.cov_re.iloc[0, 0])
print(result.scale)

# Multilevel models can also be used to estimate ICC values. In the case of a model with one level, which is the case here, the ICC is the variance of the grouping variable divided by the sum of the variance of the grouping variable and the unexplained variance, which is labeled as scale. 

icc_multilevel_model = result.cov_re.iloc[0, 0] / (result.cov_re.iloc[0, 0] + result.scale) # variance of the grouping variable / variance of the grouping variable + unexplained variance
print(icc_multilevel_model) # the ratio is around 0.014, which is similar to the estimated ICC value for the marginal model using GEE approach

	# Predicted Random Effects
	
# Remember that the actual random effects in a multilevel model are never observable, but we can predict them from the data. The predicted random effects are known as Best Linear Unbiased Predictors (BLUPs), they represent the deviation of a specific group's intercept from the overall average intercept.

print(result.random_effects) # predicted random effects (BLUPs) for the 30 groups (MVUs)

# Cluster 1241 has an unusually high BLUP, and cluster 1282 has an unusually low BLUP. BLUPs are calculated after adjusting for the covariates in the model, thus the fact that these BLUPs are still unusually high or low suggests that the variability is not explained by the covariates included in the model (i.e. the unexplained variation is captured by the random effects). 

# Must exist some characteristic for cluster 1241 (or cluster 1282) that affects systolic blood pressure, but is not captured by the covariates in the model.

	# Random Slopes Model

df['age_cen'] = df.groupby('group').RIDAGEYR.transform(lambda x: x - x.mean()) # center age variable

multilevel_model = sm.MixedLM.from_formula('BPXSY1 ~ age_cen + RIAGENDRx + BMXBMI + C(RIDRETH1)', groups = 'group', vc_formula = {'age_cen': '0+age_cen'}, data = df) # define a multilevel regression model; the vc formula argument indicates that the model should include a random slope for the age variable, note that the random intercept is already accounted for by the groups argument
result = multilevel_model.fit() # estimate model parameters
print(result.summary()) # the estimated variance for random age slopes is 0.004, which translates to a standard deviation of 0.06. 

# We can interpret the results shown above as follows: in some clusters blood pressure may increase by around 0.467 + 0.06 = 0.527 mm Hg per year, while in other clusters blood pressure may increase by only around 0.467 − 0.06 = 0.407 mm Hg per year (considering the standard deviation and the fixed effect for age).

# In the model defined above, the cluster-specific intercepts and slopes are independent random variables. Now we'll fit a model in which the cluster-specific intercepts and slopes are allowed to be correlated. 

multilevel_model = sm.MixedLM.from_formula('BPXSY1 ~ age_cen + RIAGENDRx + BMXBMI + C(RIDRETH1)', groups = 'group', re_formula = '1+age_cen', data = df) # define a multilevel regression model; re formula argument indicates that the model includes both a random intercept and a random slope for age variable and this specification allows the model to estimate a correaltion between the random intercept and the random slope
result = multilevel_model.fit() # estimate model parameters
print(result.summary())

# The estimated correlation coefficient between random slopes and random intercepts is estimated to be 0.119/\sqrt{8.655 * 0.004} which is around 0.64. This indicates that clusters with unusually high average blood pressure also tend to have blood pressure increasing faster with age.

# Note: When working with random slopes, it's common to center any covariate which has a random slope. This doesn't change the fundamental interpretation of the model, but it does often result in models that converge faster and more robustly.

