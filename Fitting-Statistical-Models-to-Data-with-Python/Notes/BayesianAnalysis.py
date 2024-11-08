# Understanding Bayesian Analysis with Python

# Bayesian statistics interprets probability as a measure of belief or certainty about an event, rather than the long-run frequency of that event (frequentist approach). Instead of providing a single best estimate, Bayesian methods yield a posterior distribution for the parameter(s) of interest. This posterior distribution combines prior beliefs about the parameter with the likelihood of the observed data, providing a range of plausible values and associated probabilities. 

# Like said above, frequentist methods interpret probabilities as long-run frequencies, and point estimates are used as fixed values with no associated distribution. 

# Coin Flipping Problem

# For a fair coin, the sample space is {heads, tails}, and each of these outcomes has probability 1/2. Now let's suppose that we have a coin and don't know whether it's fair. The actual probability of this coin landing heads-up is the parameter theta. 

# The exact value of theta can never be known, but we are able to conduct an experiment where we flip this coin n times and observe x heads and n - x tails. The data D is denoted as (x, n). 

# The likelihood p(D|theta) is a single number that represent the probability of observing the data D, for a given value of the parameter theta.
	
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
	
# This function plots the binomial likelihood function for each possible value of te parameter theta, for three given values of x, and three different sample sizes n.	

def likelihood_function(n): # sample size as function's argument
	theta = np.linspace(0.01, 0.99, 100) # array of values for theta; represents possible probabilities of success for the binomial distribution
	for x in np.floor(np.r_[0.2, 0.5, 0.6]*n): # different values of x 
		pmf = st.binom.pmf(x, n, theta) # probability mass function of the binomial distribution for each theta value
		plt.grid(True)
		plt.plot(theta, pmf, '-', label = '%.0f' % x)
		plt.xlabel(r'$\theta$', size = 12)
		plt.ylabel('Log likelihood', size = 12)
	ha, lb = plt.gca().get_legend_handles_labels()
	plt.show() # plot the likelihood values against theta, it creates a curve that shows how the likelihood changes as theta varies
	
likelihood_function(10) # no. of trials = 10
likelihood_function(100) # no. of trials = 100
likelihood_function(1000) # no. of trials = 1000

# Likelihood is higher when the proportion of heads x/n matches the parameter theta.  This is because theta values that closely match the observed proportion explain the data better. Note that the likelihood is more concentrated around x/n when the sample size n is larger.

# The posterior distribution in Bayesian statistics combines the information in the likelihood and the prior. Specifically, the posterior distribution is proportional to the likelihood times the prior. The prior reflects the knowledge about the parameter that do not come from the data. When analyzing data using Bayesian methods, we can choose a prior.  A natural choice for the prior for a binomial model is a uniform distribution on the unit interval [0, 1].

# Note: In this case, we choose a prior equal to 1, so the plots also show the posterior distribution.

# Beta Priors

# Suppose our first belief is that our coin is fair and wish to analyze the data in light of this prior knowledge. We can use a non-uniform prior that is concentrated around 1/2. A convenient way to specify a prior for a parameter such as theta, which is a probability that falls between 0 and 1, is to use a beta distribution.

alpha = 2
beta = 2 # shape parameters for beta distribution
theta = np.linspace(0.01, 0.99)
plt.grid(True)
plt.plot(theta, st.beta.pdf(theta, alpha, beta))
plt.title(r'Beta Distribution with $\alpha = \beta = 2$')
plt.xlabel(r'$\theta$', size = 15)
plt.ylabel('Prior Probabiity', size = 15)
plt.show() # this particular beta distribution places the greatest prior mass at 1/2

alpha = 1
beta = 1 # shape parameters for beta distribution
theta = np.linspace(0.01, 0.99)
plt.grid(True)
plt.plot(theta, st.beta.pdf(theta, alpha, beta))
plt.title(r'Beta Distribution with $\alpha = \beta = 2$')
plt.xlabel(r'$\theta$', size = 15)
plt.ylabel('Prior Probabiity', size = 15)
plt.show() # this particular beta distribution places the greatest prior mass at 1/2

alpha = 10
beta = 10 # shape parameters for beta distribution
theta = np.linspace(0.01, 0.99)
plt.grid(True)
plt.plot(theta, st.beta.pdf(theta, alpha, beta))
plt.title(r'Beta Distribution with $\alpha = \beta = 2$')
plt.xlabel(r'$\theta$', size = 15)
plt.ylabel('Prior Probabiity', size = 15)
plt.show() # this particular beta distribution places the greatest prior mass at 1/2

# Difference Between Bayesian and Frequentist Inference

# This function calculates the posterior distribution using a Beta prior distribution with the given values of the shape parameters alpha and beta. This function is a simplified representation of the core concept in Bayesian inference.

def posterior_distribution(x, n, alpha, beta, theta): 
    return st.binom.pmf(x, n, theta) * st.beta.pdf(theta, alpha, beta)

# In this setting, we're observing x heads out of n coin tosses and want to estimate theta, the probability of the coin landing heads-up.

x = 13 # no. of successes
n = 20 # no. of trials
plt.grid(True)
plt.axvline(x/n, color = 'black') # black horizontal line represents the MLE, which always falls at x/n; it doesn't incorporate prior beliefs or information about the parameter
alpha, beta = 10, 10 # shape parameters for beta distribution
post = posterior_distribution(x, n, 10, 10, theta) # combines the prior distribution with the likelihood of observing the data to compute a posterior value
plt.plot(theta, post, label = '10, 10')
alpha, beta = 1, 1 
post = posterior_distribution(x, n, 1, 1, theta) 
plt.plot(theta, post, label = '1, 1')
ha, lb = plt.gca().get_legend_handles_labels()
plt.xlabel(r'$\theta$', size = 12)
plt.ylabel('Posterior Probability\n(Not Normalized)', size = 12)
plt.show() 

# When alpha = beta = 1 the Beta distribution is uniform (as we saw above), so in this case the posterior distribution is influenced almost entirely by the data. Thus the peak of the posterior is very close to the MLE obtained from frequentist inference.

# On the other hand, when alpha = beta = 10 the Beta distribution is more concentrated around 0.5, indicating a strong belief that theta is close to 0.5 (a fair coin). The resulting posterior distribution is more influenced by this prior belief, and the peak of the posterior is shrunk toward 0.5, even if the data suggests a different estimate (no. of successes equal to 13). 

# In Bayesian analysis, we start with a prior belief about the parameter theta. Then we calculate the posterior distribution combining these prior beliefs with the likelihood from the observed data. Providing a refined estimate of theta that incorporates both prior information and new evidence. While frequentist methods rely solely on the observed data.

