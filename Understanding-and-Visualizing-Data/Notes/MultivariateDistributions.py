# Multivariate Distributions in Python

# The aim of this script is to show how plotting two variables together can give us information that plotting each one separately may miss

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

r = 1

mean = [15, 5]
cov = [[1, r], [r, 1]]
x, y = np.random.multivariate_normal(mean, cov, 400).T

# Plot the histograms of X and Y next to each other

plt.figure(figsize = (10, 5)) # adjust figure size

plt.subplot(1, 2, 1)
plt.hist(x = x, bins = 15) # histogram of X
plt.title("X")

plt.subplot(1, 2, 2)
plt.hist(x = y, bins = 15) # histogram of Y
plt.title("Y")

plt.show()

# Plot the data

plt.figure(figsize = (10, 10))
plt.subplot(2, 2, 2)
plt.scatter(x = x, y = y)
plt.title("Joint Distribution of X and Y")

plt.subplot(2, 2, 4)
plt.hist(x = x, bins = 15) # distribution of f_{X}(x)
plt.title("Marginal Distribution of X")

plt.subplot(2, 2, 1)
plt.hist(x = y, orientation = "horizontal", bins = 15) # distribution of f_{Y}(y)
plt.title("Marginal Distribution of Y")

plt.show()


