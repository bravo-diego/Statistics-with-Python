# The Empirical Rule and Distribution

# The empirical rule describes how many observations fall within a certain distance from our mean; the distance from the mean is denoted as sigma or standard deviation

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

random.seed(1738)

mu = 7
sd = 1.7

observations = [random.normalvariate(mu, sd) for i in range(100000)]

sns.histplot(observations)
plt.axvline(np.mean(observations) + np.std(observations), color = 'forestgreen')
plt.axvline(np.mean(observations) - np.std(observations), color = 'forestgreen')

plt.axvline(np.mean(observations) + (np.std(observations) * 2), color = 'gold')
plt.axvline(np.mean(observations) - (np.std(observations) * 2), color = 'gold')

plt.axvline(np.mean(observations) + (np.std(observations) * 3), color = 'orange')
plt.axvline(np.mean(observations) - (np.std(observations) * 3), color = 'orange')

plt.show()

print(pd.Series(observations).describe()) # summary statistics

sample_A = random.sample(observations, 100)
sample_B = random.sample(observations, 100)
sample_C = random.sample(observations, 100)

fig, ax = plt.subplots() 

sns.histplot(sample_A, ax = ax)
sns.histplot(sample_B, ax = ax)
sns.histplot(sample_C, ax = ax)

plt.show() # distributions follow the same similar trend where the mean is just about around 7 and the empirical rule still almost apply very significantly

# Empirical Distribution is a cumulative density function (cdf) that signifies the proportion of observations that are less than or equal to certain values 

ecdf = ECDF(observations)

plt.plot(ecdf.x, ecdf.y)

plt.axhline(y = 0.025, color = 'y', linestyle = '-')
plt.axvline(x = np.mean(observations) - (2 * np.std(observations)), color  = 'y', linestyle = '-')

plt.axhline(y = 0.975, color = 'y', linestyle = '-')
plt.axvline(x = np.mean(observations) + (2 * np.std(observations)), color  = 'y', linestyle = '-')

plt.show() # the distance from our mean and minus 2 standard deviations we have 2.5% of our values; the distance from our mean plus 2 standard deviations we have about 97.5% of our observations; finally within the yellow big box we have the 95% of our observations

# Basically, we use the empirical distribution function to get a better idea of at what value how many observations will we observe
