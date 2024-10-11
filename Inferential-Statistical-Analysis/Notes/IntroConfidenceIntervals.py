# Statistical Inference with Confidence Intervals

# The data are a sample from a population, and that population can be described in terms of various numerical parameters. Using the data, we can estimate a parameter of interest. For example, suppose that the parameter of interest is the mean credit card debt for all people residing in the US. We may call this (unknown) parameter s. Using a sample of data, we estimate this parameter, say using the sample mean (average) of the credit card debts for all people in our sample. We can denote this estimate by hat{s}. We know that hat{s} is not exactly equal to s, but can we somehow convey which values for s could potentially be the actual value? This is the goal of a confidence interval. 

# A confidence interval is a calculated range around a parameter estimate (a statistic) that includes all possible true values of the parameter that are consistent with the data in a certain sense. The key property of a confidence interval is that if we were to repeatedly sample from the population, calculating a confidence interval from each sample, then 95% (in a 95% confidence interval) of our calculated confidence intervals would contain ('cover') the true population.

# How are Confidence Intervals Calculated?

# A confidence interval for a population proportion can be calculated as follows:

	# Best Estimate +- Margin of Error
	
# Where the **Best Estimate** is the observed population proportion or mean and the **Margin of Error** is the t-multiplier times standard error.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# If we consider a sample of 659 people with a toddler, and 85% of these parents use a car seat all of the time. This point estimate (85%) is NOT exactly equal to the population proportion of parents who use a car seat. The standard error (SE) conveys the likely error in the point estimate relative to the population value. 

t_value = 1.96
p = 0.85
n = 659

SE = np.sqrt((p * (1 - p))/n) # standard error for proportion; how much the sample estimate differs from the true population value
print(SE) # our point estimate (85%) is likely to be around 1.4 percentage points from the truth

lower_bound = p - t_value * SE
upper_bound = p + t_value * SE
print(lower_bound, upper_bound) # confidence interval defined in terms of both lower confidence bound and upper confidence bound

print(sm.stats.proportion_confint(n * p, n)) # confidence interval using stats models module

# Confidence Interval for the Mean Cartwheel Dataset

df = pd.read_csv('/home/aspphem/Desktop/Statistics-with-Python/Cartwheeldata.csv')
print(df.head())

mean = df['CWDistance'].mean()
sd = df['CWDistance'].std()
n = len(df)

t_value = 2.064
SE = sd/np.sqrt(n) # standard error for mean

lower_bound = mean - t_value * SE
upper_bound = mean + t_value * SE
print(lower_bound, upper_bound) # confidence interval defined in terms of both lower confidence bound and upper confidence bound

print(sm.stats.DescrStatsW(df['CWDistance']).zconfint_mean()) # confidence interval using stats models module

# Confidence Intervals for the Difference Between Two Population Proportions/Means

# To illustrate comparison of population proportions, we will analyze  the difference between the proportion of females who smoke, and the proportion of male who smoke (we will use the 2015-2016 wave of the NHANES data for the analysis). To illustrate comparison of population means, we will analyze the difference between mean body mass index (BMI) for females and for males.

df = pd.read_csv('/home/aspphem/Desktop/Statistics-with-Python/nhanes_2015_2016.csv')
print(df.head())

df['SMQ020x'] = df.SMQ020.replace({1: "smoke", 2: "nosmoke", 7: np.nan, 9: np.nan}) # recoding SMQ020 variable that is coded as 1 (smoker) and 2 (non-smoker) into a new variable SMQ020x
print(df['SMQ020x'].value_counts())

df['RIAGENDRx'] = df.RIAGENDR.replace({1: "Male", 2: "Female"})
print(df['RIAGENDRx'].value_counts()) # recoding RIAGENDR that is coded as 1 (male) and 2 (female) to a new variable RIAGENDRx

dx = df[["SMQ020x", "RIAGENDRx"]].dropna()
ct = pd.crosstab(dx.RIAGENDRx, dx.SMQ020x)
print(ct)

ct["Total"] = ct["nosmoke"] + ct["smoke"]
ct["nosmoke-prop"] = ct["nosmoke"] / ct["Total"]
ct["smoke-prop"] = ct["smoke"] / ct["Total"]
print(ct) # conditional rates of smoking in females and males; sample proportions

# Constructing Confidence Intervals

	# Difference of two population proportions

# Now that we have the sample proportions of female and male smokers, we can calculate confidence intervals for the difference between the population smoking proportions. 

difference = ct.loc["Male", "smoke-prop"] - ct.loc["Female", "smoke-prop"]
print(difference) # difference between smokers proportions of males and females

# Suppose we wish to assess the precision of the estimate above (~ 0.2084). First, we assess the precision of the female and male smoking rates individually.

pf = ct.loc["Female", "smoke-prop"]
nf = ct.loc["Female", "Total"]
se_female = np.sqrt(pf * (1 - pf) / nf)
print(se_female) # standard error for hat{p}

pm = ct.loc["Male", "smoke-prop"]
nm = ct.loc["Male", "Total"]
se_male = np.sqrt(pm * (1 - pm) / nm)
print(se_male) # standar error for hat{p}

# Precisions of the female-specific and male-specific smoking rates are quite similar. The standard error of the difference between the female and male smoking rates can be obtained by pooling the standard error for females and males.

se_diff = np.sqrt(se_female**2 + se_male**2)
print(se_diff) # standard error difference between female and male smoking rates; this version of the varianze pooling rule can only be used when the two estimates being compared are independent

 # Now we can construct a 95% confidence interval for the difference between the male and female smoking rates.
 
lower_bound = difference - 1.96 * se_diff
upper_bound = difference + 1.96 * se_diff
print(lower_bound, upper_bound) # we see that any value for the proportion that falls between 0.18 and 0.23 would be compatible with the data

	# Difference of two population means

# Now we consider estimation of the mean BMI (body mass index) for females and for males, and comparing these means.

print(df["BMXBMI"].head())

summary = df.groupby("RIAGENDRx").agg({"BMXBMI": ["mean", "std", np.size]})
print(summary) # based on the table, we see that females have somewhat higher BMI than males. 

sns.boxplot(x = "RIAGENDRx", y = "BMXBMI", data = df, palette = "Blues")
plt.show() # the boxplot indicates how the two distributions overlap, while also suggesting that the female BMI distribution has a slightly higher mean and perhaps greater dispersion

sem_female = summary.loc["Female", ("BMXBMI", "std")] / np.sqrt(summary.loc["Female", ("BMXBMI", "size")]) # standard error of the mean for females and males

sem_male = summary.loc["Male", ("BMXBMI", "std")] / np.sqrt(summary.loc["Male", ("BMXBMI", "size")]) 

print(sem_female, sem_male) # the standard errrors indicate that when estimating the population mean BMI values for females and for males, the female estimate will be slightly less precise; this reduced precision is largerly due to the greater internal variability of the female BMI values

sem_diff = np.sqrt(sem_female**2 + sem_male**2)
print(sem_diff)

difference = summary.loc["Female", ("BMXBMI", "mean")] - summary.loc["Male", ("BMXBMI", "mean")]

lower_bound = difference - 1.96 * sem_diff
upper_bound = difference + 1.96 * sem_diff

print(lower_bound, upper_bound) # we see that any value for the proportion that falls between 0.8 and 1.53 would be compatible with the data

