# Sampling Distributions

# Using the NHANES data to explore the sampling distributions of statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.distributions import norm

df = pd.read_csv("/home/aspphem/Desktop/StatisticsPythonCourse/Datasets/nhanes_2015_2016.csv")

# Sampling distribution of the mean 

# Sampling distributions describe how the value of a statistic computed from data varies when repeated samples of data are obtained. This can be explored mathematically, or by using a computer to simulate data repeatedly from a hypothetical population. When working with non-simulated data (e.g. NHANES data), we usually do not have the ability to explicitly obtain an independent copy of the sample to actually see its sampling distribution. However we can subsample from a dataset to mimic what would happen if we were to sample repeatedly from the population that produced it. A subsample is a random sample drawn form a larger data set, containing only a fraction of its observations

m = 100 # subsample size
subsample_diff = [] # list to storage our subsample mean differences

for i in range(1000):
    dx = df.sample(2*m) # two subsamples of size m 
    dx1 = dx.iloc[0:m, :]
    dx2 = dx.iloc[m:, :] 
    subsample_diff.append(dx1.BPXSY1.mean() - dx2.BPXSY1.mean()) # difference of mean BPXSY1 values
    
sns.histplot(subsample_diff)
plt.axvline(x = np.mean(subsample_diff), color = 'r')
plt.show() 

print(pd.Series(subsample_diff).describe()) # mean of systolic blood pressures calculated for two samples each with 100 people will typically by around 2.8 mm/Hg (standard deviation) 

# The sample size is a major determinant of the chance fluctuations in any statistic 

m = 400 # subsample size
subsample_diff = [] # list to storage our subsample mean differences

for i in range(1000):
    dx = df.sample(2*m) # two subsamples of size m 
    dx1 = dx.iloc[0:m, :]
    dx2 = dx.iloc[m:, :] 
    subsample_diff.append(dx1.BPXSY1.mean() - dx2.BPXSY1.mean()) # difference of mean BPXSY1 values
    
sns.histplot(subsample_diff)
plt.axvline(x = np.mean(subsample_diff), color = 'y')
plt.show() 

print(pd.Series(subsample_diff).describe()) # the standard deviation is around 1.38, which is close to half of what it was when we used samples of size 100; chance fluctuations in the mean systolic blood pressure are smaller when we have a larger sample size. We are able to estimate the population mean systolic blood pressure with more precision when we have samples of size 400 compared to when we have samples of size 100

# N O T E: increasing the sample size by a factor of 4 (from 100 to 400) led to a reduction of the standard deviation by a factor of 2; this scaling behavior is very common in statistics (increasing the sample size by a factor of K leads to a reduction in the standard deviation by a factor of sqrt(K))

sns.histplot(df.BPXSY1.dropna(), kde = True)
plt.show()

m = 50
subsample_mean = []
for i in range(1000):
    dx = df.sample(m)
    subsample_mean.append(dx.BPXSY1.dropna().mean())

sns.histplot(subsample_mean, kde = True, stat = 'probability')

x = np.linspace(np.min(subsample_mean), np.max(subsample_mean), 100)
y = norm.pdf(x, np.mean(subsample_mean), np.std(subsample_mean))
plt.plot(x, y, color = 'orange')

plt.show() # the distribution of means is also approximately normal, as shown by the orange curve, which is the best-fitting normal approximation to the data
