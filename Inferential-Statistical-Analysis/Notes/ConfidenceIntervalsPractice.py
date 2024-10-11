# Confidence Intervals Practice using NHANES Dataset

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt

df = pd.read_csv('/home/aspphem/Desktop/Statistics-with-Python/nhanes_2015_2016.csv')
print(df.head())

# Confidence intervals for one proportion

# We will calculate the proportions of smokers separetely for females and males. Initially we can compare these two proportions and their corresponding confidence intervals informally, then we can compare two proportions formally using confidence intervals.

df['SMQ020x'] = df.SMQ020.replace({1: "Yes", 2: "No", 7: np.nan, 9: np.nan}) 
df['RIAGENDRx'] = df.RIAGENDR.replace({1: "Male", 2: "Female"}) # replace numeric codes in the variables of interest with text labels

dx = df[["SMQ020x", "RIAGENDRx"]].dropna()
print(pd.crosstab(dx.SMQ020x, dx.RIAGENDRx))

# The confidence interval (CI) is constructed using two inputs: the sample proportion, and the total sample size; in order to construct a confidence interval for a population proportion, we must be able to assume the sample proportions follow a normal distribution.

dz = dx.groupby(dx.RIAGENDRx).agg({"SMQ020x": [lambda x: np.mean(x == "Yes"), np.size]})
dz.columns = ["Proportion", "N"]
print(dz)

# Confidence intervals are closely connected to standard errors. The standard error essentially tells us how far we should expect an estimate to fall from the truth. A confidence interval is an interval that under repeated sampling covers the truth a definded proportion of the time, this coverage probability could be set to 90, 95 or 99%. It turns out that in many things, a 95% confidence interval can be constructed as the interval consisting of all points that are within two standard erros of the point estimate. More concisely, the confidence interval approximately spans from hat{theta} - 2 * SE to hat{theta} + 2 * SE, where hat{theta} is the point estimate and SE is the standar error.

p = dz.Proportion.Female # female proportion 
n = dz.N.Female # number of females 
standard_error_females = np.sqrt(p * (1 - p)/n)
print(standard_error_females)

lower_bound = p - 1.96 * np.sqrt(p * (1 - p)/n)
upper_bound = p + 1.96 * np.sqrt(p * (1 - p)/n)

print("({}, {}".format(lower_bound, upper_bound)) # confidence interval for the proportion of female smokers

p = dz.Proportion.Male # male proportion 
n = dz.N.Male # number of males 
standard_error_males = np.sqrt(p * (1 - p)/n)
print(standard_error_males)

lower_bound = p - 1.96 * np.sqrt(p * (1 - p)/n)
upper_bound = p + 1.96 * np.sqrt(p * (1 - p)/n)

print("({}, {}".format(lower_bound, upper_bound)) # confidence interval for the proportion of male smokers

# Using statsmodels library to calculate the confidence intervals for both female and male smokers proportions

print(sm.stats.proportion_confint(906, 2972)) # (no. of smokers, sample size)

print(sm.stats.proportion_confint(1413, 2753))

# Confidence intervals comparing two independent proportions

# The confidence intervals for the proportion of female and male smokers shown above are quite narrow and don't overlap; this suggests that there is a substantial difference between the lifetime smoking rates for women and men. However there is no explicit information here about how different the two population proportions might be. To address this question, we can form a confidence interval for the difference between the proportions of females who smoke and the proportion of males who smoke.

# The point estimate of the difference between female and male smoking rates is -0.208 (i.e. 0.305 - 0.513, proportions shown above); that is, the smoking rate is about 20% point higher in men than in women. This difference of arounf 20% points is only a point estimate of the true value, it is NOT exactly equal to the difference between the unknown proportions of females and males who smoke in the population. A confidence interval helps us assess how far the estimated difference may be from the true difference. 

# The difference between two sample proportions based on independent data has a standard error that reflects the combined uncertainty in the two proportions being differenced.

standard_error_difference = np.sqrt(standard_error_females**2 + standard_error_males**2) # this formula is only accurate if the two sample proportions being differenced are based on independent samples
print(standard_error_difference) # the standard error indicates that the estimated difference statistic -0.208 is expected to fall around 0.013 units from the true value; we don't know in which direction the error lies, and we don't know that the error is exactly 0.013, only that it's around this size on average

difference = dz.Proportion.Female - dz.Proportion.Male
lower_bound = difference - 2*standard_error_difference
upper_bound = difference + 2*standard_error_difference

print(difference) # proportion difference around -0.208
print("({}, {})".format(lower_bound, upper_bound)) # 95% confidence interval; any value for the difference of population proportions (between females and males) lying between -0.233 and -0.183 is consistent with the observed data

