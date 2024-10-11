# Hypothesis Testing Practice using NHANES Dataset

import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import scipy.stats.distributions as dist

path = '/home/aspphem/Desktop/Statistics-with-Python/nhanes_2015_2016.csv' # path file
df = pd.read_csv(path) # read csv file
print(df.head()) # data preview
print(df.keys()) # column names

df['SMQ020x'] = df.SMQ020.replace({1: 'Yes', 2: 'No', 7: np.nan, 9: np.nan}) 
df['RIAGENDRx'] = df.RIAGENDR.replace({1: 'Male', 2: 'Female'}) # convert integer codes to text values

# Hypothesis Test for One Proportion

# One-sample test that the population proportion of smokers is 0.4 (H_{0}: p = 0.4).

x = df.SMQ020x.dropna() == 'Yes' # smokers in the sample
p = np.mean(x) # smokers proportion
print("Proportion of smokers in the US: {:.2f}".format(p))

se = np.sqrt((0.4 * (1 - 0.4))/len(x)) # standard error
print("Standard error {}".format(se))

test_stat = (p - 0.4)/se # z test statistic
print("Test statistic {}".format(test_stat))

p_value = (1 - dist.norm.cdf(test_stat)) * 2 # p-value
print("p-value {}".format(p_value)) # p-value > 0.05 thus we fail at reject the null hypothesis; NHANES data are compatible with the proportion of smokers in the US being 40%.

# Using statsmodels library to perform the test.

print(sm.stats.proportions_ztest(x.sum(), len(x), 0.4)) # sample proportion is used to estimate the standard error (SE)
print(sm.stats.proportions_ztest(x.sum(), len(x), 0.4, prop_var=0.4)) # value of the proportion under the null hypothesis is used to estimate the standard error (SE)

print(sm.stats.binom_test(x.sum(), len(x), 0.4)) # binomial p-value; parameters: no. of successes, no. of trials, probability of success

# Hypothesis Test for Two Proportions

# Compare the smoking rates between females and males. Since smoking rates vary strongly with age, we do this in the subpopulation of people between 20 and 25 years of age.

dx = df[['SMQ020x', 'RIDAGEYR', 'RIAGENDRx']].dropna()
p = dx.groupby('RIAGENDRx')['SMQ020x'].agg([lambda z: np.mean(z == 'Yes'), 'size']) 
p.columns = ['Smoke', 'N']
print(p) # proportion of smokers by gender

p_smoke = (dx.SMQ020x == 'Yes').mean() # smokers proportion
var = p_smoke * (1 - p_smoke) # variance
se = np.sqrt(var * (1 / p.N.Female + 1 / p.N.Male)) # standard error

test_stat = (p.Smoke.Female - p.Smoke.Male) / se # z test statistic
print("Test statistic {}".format(test_stat))

p_value = (dist.norm.cdf(test_stat)) * 2 # p-value
print("p-value {}".format(p_value)) # p-value < 0.05 thus we have strong evidence against the null hypothesis; the difference between smoking rates by gender is statistically significant

# Hypothesis Test Comparing Means

# Compare systolic blood pressure to the fixed value 120 (which is the lower threshold for "pre-hypertension").

dx = df[['BPXSY1', 'RIDAGEYR', 'RIAGENDRx']].dropna()
dx = dx.loc[(dx.RIDAGEYR >= 40) & (dx.RIDAGEYR <= 50) & (dx.RIAGENDRx == 'Male'), :] # systolic blood preasure values for males around 40-50 years old
print(dx.BPXSY1.mean()) # systolic blood preasure mean for males

# Using statsmodels library to perform the test.

print(sm.stats.ztest(dx.BPXSY1, value = 120)) # sample; mean value under the null hypothesis

# Since p-value < 0.05, we can say that mean is significantly different from 120.

# Difference in Means for Independent Groups

# Test the null hypothesis that the mean blood pressure for women between the ages of 50 and 60 is equal to the mean blood pressure of men between the ages of 50 and 60.

dx = df[['BPXSY1', 'RIDAGEYR', 'RIAGENDRx']].dropna()
dx = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :] # systolic blood preasure values for both males and females around 50-60 years old
print(dx.head())

bpx_female = dx.loc[dx.RIAGENDRx == 'Female', 'BPXSY1'] # systolic blood preasure values for females
bpx_male = dx.loc[dx.RIAGENDRx == 'Male', 'BPXSY1'] # systolic blood preasure values for males
print("Systolic blood preasure mean values: females {:.2f}; males {:.2f}".format(np.mean(bpx_female), np.mean(bpx_male)))

# Using statsmodels library to perform the test.

print(sm.stats.ztest(bpx_female, bpx_male)) # z-test

# Since p-value > 0.05, we can say that the difference between mean values is not statistically significant. 

print(sm.stats.ttest_ind(bpx_female, bpx_male)) # t-test for the means of two independent samples of scores; this test assumes that the populations have identical variances by default

# Note - When the sample size is large, the difference between the t-test and z-test is very small.

# Paired Tests

# A paired t-test for means is equivalent to taking the difference between the 1st and 2nd measurement, and using a one-sample test to compare the mean of these differences to zero. 

dx = df[["BPXSY1", "BPXSY2"]].dropna() # systolic blood pressure is measured at least two times
db = dx.BPXSY1 - dx.BPXSY2 # difference between the first and second measurement
print(db.mean()) # mean value of difference between the first and second measurements

# Using statsmodels library to perform the test.

sm.stats.ztest(db) # p-value < 0.05 thus we have strong evidence against the null hypothesis; there is strong evidence that the mean values for the first and second blood pressure measurement differ

# 95% Confidence Interval

mean = np.mean(db) # mean value
sd = np.std(db) # standard deviation value
n = len(db)

SE = sd/np.sqrt(n) # standard error for mean
t = 1.96 # 95% confidence with n > 1000

lower_bound = mean - t * SE
upper_bound = mean + t * SE
print(lower_bound, upper_bound) # confidence interval defined in terms of both lower confidence bound and upper confidence bound

print(sm.stats.DescrStatsW(db).zconfint_mean()) # confidence interval using stats models module

# We see that any value for the mean difference that falls between 0.54 and 0.80 would be compatible with the data. Since 0 value is NOT included in the interval, this indicates that the mean difference is statistically significant. This conclusion aligns with our findings from hypothesis testing, where the p-value is less than 0.05.

# Power and Sample Size for Hypothesis Tests

# The term 'statistical power' refers to the probability of correctly rejecting the null hypothesis when it is false, it measures the test's ability to detect a true effect.

dx = df[["RIAGENDRx", "BPXSY1", "BPXSY2", "RIDAGEYR"]].dropna()

p_values = []
dy = dx.loc[(dx.RIDAGEYR >= 50) & (dx.RIDAGEYR <= 60), :] # select individuals aged around 50-60
subsamples = [100, 200, 400, 800] # samples sizes
for sample in subsamples:
    p_value = []
    for i in range(500): # for each sample size the paired z-test is repeated 500 times
        dz = dy.sample(sample)
        db = dz.BPXSY1 - dz.BPXSY2
        _, p = sm.stats.ztest(db.values, value = 0)
        p_value.append(p)
    p_value = np.asarray(p_value)
    p_values.append(p_value)
    print((p_value <= 0.05).mean()) # print the proportion of trials (out of the 500) that yield a p-value smaller than 0.05

# The power is approximately 0.2 when the sample size is 100, i.e. only about 20% of trials can detect a significant difference. As the sample size increases, the statistical power approaches 1.0, leading to a greater number of trials detecting a significant difference.

# All subsamples come from the same population, meaning the underlying means don't change. The only variable affecting the detection of differences is the sample size. There is a strong relationship between the sample size and the behavior of a hypothesis test. 

sns.displot(p_values[0])
plt.show() # distribution of p-values for 500 subsamples of data size 100; there are more p-values greater than 0.05

sns.displot(p_values[3])
plt.show() # distribution of p-values for 500 subsamples of data size 800; p-values are much more concentraetd close to zero 

# Disclaimer: NHANES data are a "complex survey". The data are not an independent and representative sample from the target population. Proper analysis of complex survey data should make use of additional information about how the data were collected. Since complex survey analysis is a somewhat specialized topic, we ignore this aspect of the data here, and analyze the NHANES data as if it were an independent and identically distributed sample from a population.

