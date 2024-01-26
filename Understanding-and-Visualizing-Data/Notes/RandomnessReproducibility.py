# Randomness and Reproducibility 

# If we have complete randomness, our estimates of means, proportions, and totals are unbiased; this means our estimates are equal to the population values on average

# In python we refer to randomness as the ability to generate data, strings, or more generally, numbers at random. However, when conducting analysis it is important to consider reproducibility. If we are creating random data, how can we enable reproducible analysis? 

# We do this by using pseudo-random number generators; pseudo-random number generators start with a random number, known as seed, and them use an algorithm to generate a pseudo-random sequence based on it; this means we can replicate the output  of a random number generator in python simply by knowing which seed was used 

import random
import numpy as np

random.seed(1234) # setting a seed
print(random.random())

print(random.uniform(25, 50)) # random number from uniform distribution

unif_numbers = [random.uniform(8, 10) for n in range(5)]
print(unif_numbers)

print(random.normalvariate(0, 1)) # random number from normal distribution

norm_numbers = [random.normalvariate(0, 1) for n in range(5)]
print(norm_numbers)

# Random sampling from a population

# Simple random sampling has the following properties: 1) start with known list of N population units and random select n units from the list; 2) every unit has equal probability of selection (n/N); 3) all possible samples of size n are equaly likely; 4) estimates of means, proportions, and totals based on simple random sampling are unbiased (i.e. they are equal to the population values on average)

population = [random.normalvariate(0, 1) for n in range(10000)]

sample_A = random.sample(population, 500)
sample_B = random.sample(population, 500)

print("\nMean sample A: {} \nMean sample B: {}".format(np.mean(sample_A), np.mean(sample_B)))

print("\nStandard deviation sample A: {} \nStandard deviation sample B: {} ".format(np.std(sample_A), np.std(sample_B)))

print("\n{} {}".format(np.mean(population), np.std(population))) # population mean and standard deviation 
