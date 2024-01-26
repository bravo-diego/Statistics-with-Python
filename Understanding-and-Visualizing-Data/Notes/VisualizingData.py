# How to perform some basic visualizations in Python and explore data from a graphical perspective. 

import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")

print(tips.head(5)) # show the first 5 rows of the data
print(tips.describe()) # summary statistics for the quantitative variables

sns.histplot(tips["total_bill"], kde = False).set_title("Histogram of Total Bill") # distribution of 'total bill' column

plt.show()

sns.histplot(tips["tip"], kde = False).set_title("Histogram of Total Tip") # distribution of 'tip' column

plt.show()

sns.histplot(tips["total_bill"], kde = False).set_title("Histogram of Total Bill") 
sns.histplot(tips["tip"], kde = False).set_title("Histogram of Total Tip") 

plt.show() # plot both total bill and tips histograms

sns.boxplot(tips["total_bill"]).set_title("Box plot of Total Bill") # boxplots don't show the shape of the distribution, but they can give us a better idea about the center and spread of the distribution as well as any potential outliers that may exist

plt.show()

sns.boxplot(tips["tip"]).set_title("Box plot of Tip") 

plt.show()

sns.boxplot(x = tips["tip"], y = tips["smoker"]) # boxplot of the tips grouped by smoking status

plt.show()

sns.boxplot(x = tips["tip"], y = tips["day"]) # boxplot of the tips grouped by day
g = sns.FacetGrid(tips, row = "day")
g = g.map(plt.hist, "tip") # histogram of the tips grouped by day 

plt.show()
