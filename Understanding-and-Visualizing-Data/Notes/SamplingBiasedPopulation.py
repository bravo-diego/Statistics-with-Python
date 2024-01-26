# This script looks at a hypothetical problem that illustrates what happens when we sample from a biased population and not the entire population we are interested in

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulation

mean_uofm = 155
sd_uofm = 5
mean_gym = 185
sd_gym = 5
gymperc = 0.3
population = 40000

# Create two subgroups

students = np.random.normal(mean_uofm, sd_uofm, int(population * (1 - gymperc))) # draw a random sample from a normal distribution; (mean, standard deviation, size)
students_gym = np.random.normal(mean_gym, sd_gym, int(population * (gymperc)))

total_population = np.append(students, students_gym) # create the population from the subgroups 

# Set up the figure for plotting

plt.figure(figsize = (10, 12))

plt.subplot(3, 1, 1) # students only
sns.histplot(students)
plt.title("UofM Students Only")
plt.xlim([140, 200])

plt.subplot(3, 1, 2) # gym goers only
sns.histplot(students_gym)
plt.title("Gym Goers Only")
plt.xlim([140, 200])

plt.subplot(3, 1, 3) # total population
sns.histplot(total_population)
plt.title("Full Population of UofM Students")
plt.axvline(x = np.mean(total_population), c = 'r')
plt.xlim([140, 200])

plt.show()

# What happens if we sample from the entire population?

# Simulation parameters

samples = 5000
sample_size = 50

# Get the sampling distribution of the mean from total population

mean_distribution = np.empty(samples)
for i in range(samples):
    random_students = np.random.choice(total_population, sample_size)
    mean_distribution[i] = np.mean(random_students)
    
plt.figure(figsize = (10, 8))

plt.subplot(2, 1, 1) # total population
sns.histplot(total_population)
plt.title("Full Population of UofM Students")
plt.axvline(x = np.mean(total_population), c = 'r')
plt.xlim([140, 200])

plt.subplot(2, 1, 2) # sampling distribution
sns.histplot(mean_distribution)
plt.title("Sampling Distribution of the Mean of all UofM Students")
plt.axvline(x = np.mean(total_population), c = 'k')
plt.axvline(x = np.mean(mean_distribution), c = 'r')
plt.xlim([140, 200])

plt.show() # mean of sampling distribution falls almost exactly on top of the mean of the population; if I were to repeat this process of just sampling 50 random students, and calculate the mean, I should expect on average to get a mean for my sample that is almost exactly equal to my population if I sample truly from everyone in my population, if I truly take samples from every one of my population

# We get means that are roughly equivalent to what we hope they are actually trying to measure

# What happens if we take a non-representative sample? 

samples = 5000
sample_size = 3

mean_distribution = np.empty(samples)
for i in range(samples):
    random_students = np.random.choice(students_gym, sample_size)
    mean_distribution[i] = np.mean(random_students)
    
plt.figure(figsize = (10, 8))

plt.subplot(2, 1, 1) # total population
sns.histplot(total_population)
plt.title("Full Population of UofM Students")
plt.axvline(x = np.mean(total_population), c = 'r')
plt.xlim([140, 200])

plt.subplot(2, 1, 2) # sampling distribution
sns.histplot(mean_distribution)
plt.title("Sampling Distribution of the Mean of Gym Goers")
plt.axvline(x = np.mean(total_population), c = 'k')
plt.axvline(x = np.mean(students_gym), c = 'r')
plt.xlim([140, 200])

plt.show() # in this case we only measured people at the gym, and as a result, we got an incredibly biased estimate of the full population; this is some of the dangers of using things like convenience samples

# Sometimes you are not taking samples of the entire population but instead of only select subsets, and as a result we can get really biased estimates
