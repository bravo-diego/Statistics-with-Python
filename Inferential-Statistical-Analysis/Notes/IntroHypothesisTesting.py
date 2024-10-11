# Introduction to Hypothesis Testing in Python

# With any hypotheses test, first we need to start with the hypothesis even before collect any data so we are not will be influenced by the data. 

# The first hypothesis is called 'null hypothesis' and the second hypothesis is going to be the 'alternative hypothesis' (e.g. H_{0}: p = 0.52 and H_{a}: p ? 0.52). Depending of the context H_{a} could be less than, greater than or not equal to p. Finally we set the significance level (alpha value) which typically is 0.05. The alpha value is basically the cut-off point of when we've found something to be significant.

# Then we check some assumptions: 1) first we need a random sample (every member of the population has an equal chance of being selected); 2) then we need to check if our sample size is large enough to ensure that our sample proportions follows a normal distribution.

# Difference Between One-Sided and Two-Sided Tests

# One-Sided Test: A one-sided test is used when we are specifically testing for an effect in a particular direction, either greater than or less than a certain value. 

# Two-Sided Test: A two-sided test is used when the research hypothesis is non-directional, meaning we are interested in whether a parameter is simply different from a certain value, regardless of the direction of the difference.

# Testing a Difference in Population Proportions

# In hypothesis testing for a difference in population proportions, a two-sided test is used when we're interested in testing whether the two population proportions are different, without specifying the direction of the difference, i.e.

	# H_{0}: p_{1} = p_{2} or p_{1} - p_{2} = 0 

	# H_{a}: p_{1} ≠ p_{2} or p_{1} - p_{2} ≠ 0 
	
# Is called two-sided because we are testing whether p_{1}​ could be either greater than or less than p_{2}​, we're looking for any significant difference regardless of the direction.

# Since is a two-sided test we want to check for significant deviations in either direction, thus the p-value is calculated as:

	# p-value = 2 * P(Z > |Z_{observed}|)

# If the p-value is less than your significance level (i.e. alpha < 0.05), reject the null hypothesis. This means there is sufficient evidence to suggest that p_{1} ≠ p_{2}​. Otherwise (i.e. alpha > 0.05) we fail to reject the null hypothesis, meaning there's not enough evidence to suggest a difference between the proportions.

# Note: having a really low p-value could mean statistical significance, but it doesn't necessarily mean practical significance.

# The theory about hypothesis testing defines two types of errors: a type I error (false positive) and a type II error. The type I error occurs when the null hypothesis is true but is incorrectly rejected. A type II error occurs when the null hypothesis is not rejected when it actually is false. Most methods for statistical inference aim to strictly control the probability of a type I error, usually at 5%.

# Testing Theories about Population Parameters in Cartwheel Dataset

# Research Question: Is the average cartwheel distance for adults more than 80 inches?

# Define Null and Alternative Hypotheses:

	# H_{0}: population mean CW distance is 80 inches (H_{0}: mu = 80)
	
	# H_{a}: population mean CW distance is greater than 80 inches (H_{0}: mu > 80)

# Set out significance level for the test: Standard significance level 5%

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

from scipy.stats import t

null_value = 80

df = pd.read_csv('/home/aspphem/Desktop/Statistics-with-Python/Cartwheeldata.csv') 
print(df.head())

print(df.describe()['CWDistance']) # summarize data of CWDistance column

# Graphical Summaries

plt.hist(df['CWDistance']) 
plt.title('Histogram of CW Distances')
plt.show()

fig = sm.qqplot(df['CWDistance'], fit = True, line = 'r')
plt.title('QQ Plot for CW Distance')
plt.show() # both histogram and QQ plots suggest deviations from normality

# How Compatible is the Data with the Null Hypothesis?

sample_mean = np.mean(df['CWDistance'])
print(sample_mean) # sample mean

sample_std = np.std(df['CWDistance'])
print(sample_std) # sample standard deviation

estimated_se = sample_std/np.sqrt(len(df['CWDistance'])) # sample standard deviation s / sqrt(n)
print(estimated_se) # estimated standard error

t_stat = (sample_mean - null_value) / estimated_se # best estimate - null value / estimated standard error
print(t_stat) # test statistic

# Our sample mean is only 0.84 standard errors above null value of 80 inches.

# Determine p-value

# p-value: Probability of seeing test statistic of 0.84 or more extreme assuming the null hypothesis is true

dof = 24 # degrees of freedom (n - 1)

p_val = (1 - t.cdf(abs(t_stat), dof))
print(p_val) # p-value

# Sice our p-value is greater than the significance level 0.05, we cannot reject the null hypothesis (weak evidence against null hypothesis). Based on estimated mean we cannot support the population mean CW distance is greater than 80 inches. 

