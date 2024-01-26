# In this document, we are going to outline the most common uses for each of the following libraries. 

import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

    # NumPy
    
# A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the rank of the array; the shape of an array is a tuple of integers giving the size of the array along each dimension.

a = np.array([1, 2, 3]) # 3x1 numpy array

print(type(a)) 

print(a.shape)

print(a[0], a[1], a[2]) # print all values in a

b = np.array([[1, 2], [3, 4]]) # 2x2 numpy array

print(b.shape)

print(b[0, 0], b[0, 1], b[1, 0], b[1, 1]) # print all values in b

c = np.array([[1, 2], [3, 4], [5, 6]]) # 3x2 numpy array

print(c.shape)

print(c[0, 0], c[1, 0], c[2, 0]) # print some values in c

d = np.zeros((2, 3)) # 2x3 numpy array with zeros in each entry

print(d)

e = np.ones((4, 2)) # 4x2 numpy array with ones in each entry

print(e)

f = np.full((2,2), 10) # 2x2 numpy array with constant value (10)

print(f)

g = np.random.random((3, 3)) # 3x3 numpy array with random values

print(g)

# Array Indexing

h = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])

print(h)

i = h[:2, 1:3]
j = h[1:, 2:]

print(i, "\n", j)

# Datatypes in Arrays

k = np.array([1, 2])
print(k.dtype) # integer

l = np.array([1.0, 2.0])
print(l.dtype) # float

m = np.array([1.0, 2.0], dtype = np.int64)
print(m.dtype) # force data type to integer

# Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module.

x = np.array([[1, 2], [3, 4]], dtype=np.float64)
y = np.array([[5, 6], [7, 8]], dtype=np.float64)

print(x + y) # addition
print(np.add(x, y))

print(x - y) # difference
print(np.subtract(x, y))

print(x * y) # product
print(np.multiply(x, y))
print(stats.norm.cdf(np.array([1, -1., 0, 1, 3, 4, -2, 6])))
print(x / y) # division
print(np.divide(x, y))

print(np.sqrt(x)) # square root 

print(np.sum(x)) # sum of all elements

print(np.sum(x, axis = 0)) # sum of each column
print(np.sum(x, axis = 1)) # sum of each row

print(np.mean(x))

    # SciPy
    
# The SciPy.Stats module contains a large number of probability distributions as well as a growing library of statistical functions.

print(stats.norm.rvs(size = 10)) # sample (n = 10) of a normal random variable 

print(stats.norm.cdf(np.array([1, -1., 0, 1, 3, 4, -2, 6]))) # cumulative distribution function of a normal random variable

# Descriptive Statistics

np.random.seed(282629734)

x = stats.t.rvs(10, size = 1000) # generate 1000 student's T continuous random variables

print(x.min())
print(x.max())
print(x.mean())
print(x.var()) # all are equivalent to np.method.(x)

print(stats.describe(x))

    # Matplotlib
    
# Matplotlib is a plotting library. 

x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

plt.plot(x, y) # plot the points using matplotlib
plt.show() # show the figure

y_sin = np.sin(x)
y_cos = np.cos(x)

plt.plot(x, y_sin) # plot the points using matplotlib
plt.plot(x, y_cos)
plt.xlabel("x axis")
plt.ylabel("y axis")
plt.title("Sine and Cosine")
plt.legend(["Sin", "Cos"])

plt.show() # show the figure

# Subplots

plt.subplot(2, 1, 1) # subplot grid of height 2 and width 1; first plot
plt.plot(x, y_sin) 
plt.title("Sine")

plt.subplot(2, 1, 2) # second plot
plt.plot(x, y_cos) 
plt.title("Cosine")

plt.show()

    # Seaborn
    
# Seaborn is complementary to Matplotlib and it specifically targets statistical data visualization. 

# Scatterplots

df = pd.read_csv("/home/aspphem/Desktop/StatisticsPythonCourse/Datasets/Cartwheeldata.csv") # importing data via pandas library

sns.lmplot(x = "Wingspan", y = "CWDistance", data = df, fit_reg = False, hue = 'Gender') # scatterplot; no regression line; color by evolution stage

plt.show()

sns.swarmplot(x = "Gender", y = "CWDistance", data = df) 

plt.show()

# Boxplots

sns.boxplot(data = df.loc[:, ["Age", "Height", "Wingspan", "CWDistance", "Score"]])

plt.show()

sns.boxplot(data = df.loc[df['Gender'] == 'M', ["Age", "Height", "Wingspan", "CWDistance", "Score"]]) # male boxplot

plt.show()

sns.boxplot(data = df.loc[df['Gender'] == 'F', ["Age", "Height", "Wingspan", "CWDistance", "Score"]]) # female boxplot

plt.show()

# Histogram

sns.distplot(df.CWDistance) # distribution plot

plt.show()

# Count Plot

sns.countplot(x = 'Gender', data = df)
plt.xticks(rotation = -45)

plt.show()
