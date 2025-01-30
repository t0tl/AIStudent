# Week 2: Statistics

## Resources

- The slides are available [here](slides/build/main.pdf).
- The recording is available [here](https://drive.google.com/file/d/1njewMOkq7vgHjy18Nopfx2vb14DRTVP5/view?usp=sharing).

## Installing Required Packages with `pip`

This week, we will be using the following packages for our exercises:

- `numpy`: For numerical computations
- `pandas`: For data manipulation and analysis
- `matplotlib`: For data visualization
- `scikit-learn`: For machine learning algorithms
- `scipy`: For scientific computing

You can install these packages using `pip` by running the following command in your terminal or command prompt:

```bash
pip install numpy pandas matplotlib scikit-learn scipy
```

If this command doesn't work, you can try using `pip3` instead of `pip`:

```bash
pip3 install numpy pandas matplotlib scikit-learn scipy
```

## Exercise 1: Analyzing (Descriptive Statistics + Data Visualization)

### Contents

Mean, Median, Mode, Variance, Standard Deviation, Range, Histograms, Box Plots, Scatter Plots, Bar charts

### Story

You have been given a dataset containing information about students' test scores. Your task is to analyze the data and provide insights to help improve student performance.

### Tasks

- Load the dataset
- Compute descriptive statistics (mean, median, mode, variance, standard deviation, range)
- Visualize the data using histograms, box plots, scatter plots, and bar charts
- Explain the insights you gained from the analysis

### Starting Point

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/Student_Scores_Data.csv')

# Display the first few rows of the data
print(data.head())

# Compute descriptive statistics

###################################
# Compute the statistics here     #
# #### TODO ####                  #
###################################

# Check out numpy's functions for calculating these statistics
# Here: https://numpy.org/doc/stable/reference/routines.statistics.html#averages-and-variances

# Visualize the data using histograms, box plots, scatter plots, and bar charts

###################################
# Visualize the data here         #
# #### TODO ####                  #
###################################

# Check out matplotlib's functions for creating these visualizations
# Here: https://matplotlib.org/stable/gallery/index.html
```

## Exercise 2: Probability (Probability Theory + Simulation)

### Topics

Probability, Conditional Probability, Bayes' Theorem, Random Variables, Probability Distributions, Simulation

### Tasks

- Simulate 1000 coin tosses, calculate the probability of getting heads, and compare it with the theoretical probability.
- Simulate 1000 dice rolls, calculate the probability of getting a prime number, and compare it with the theoretical probability. Calculate the conditional probability of getting a prime number given that the number is odd.
- Use Monte Carlo simulation to estimate the value of $\pi$ by generating random points in a unit square and calculating the ratio of points inside a quarter circle to the total points.

#### Task 1: Coin Toss Simulation

```python
# Simulate 1000 coin tosses
coin_tosses = np.random.choice(['H', 'T'], size=1000)

# Calculate the probability of getting heads
# Theoretical probability: 0.5
# probability_heads = ...
# #### TODO ####

print(f"Probability of getting heads: {probability_heads}")
```

#### Task 2: Dice Roll Simulation

```python
# Simulate 1000 dice rolls
dice_rolls = np.random.randint(1, 7, size=1000)

# Calculate the probability of getting a prime number
# Theoretical probability: p(prime) = 
# probability_prime = ...
# #### TODO ####

print(f"Probability of getting a prime number: {probability_prime}")

# Calculate the conditional probability of getting a prime number given that the number is odd
# Theoretical probability: P(prime | odd) = P(prime and odd) / P(odd) = 
# probability_prime_given_odd = ...
# #### TODO ####

print(f"Conditional probability of getting a prime number given that the number is odd: {probability_prime_given_odd}")
```

#### Task 3: Monte Carlo Simulation for Estimating Pi

```python
# Monte Carlo simulation to estimate the value of pi
n = 1000000 # Try with different values of n!!!
points = np.random.rand(n, 2) # Generate random points in a unit square, each point is (x, y)
# pi_estimate = ...
# #### TODO ####

# Remember what we learned about the area of a quarter circle and a square
# pi = 4 * (area of quarter circle) / (area of square)

###############################################################
# You need to see which points are inside the quarter circle! #
# #### TODO ####                                              #
###############################################################

print(f"Estimated value of pi: {pi_estimate}") # Should be close to 3.14159...
```

## Exercise 3: Correlation and Regression (Correlation + Regression Analysis)

### Story

You work for a car dealership, and your task is to analyze the relationship between car prices and their mileage. You want to determine if there is a correlation between the two variables and build a regression model to predict car prices based on mileage.

### Tasks

- Calculate the correlation coefficient between car prices and mileage.
- Build a simple linear regression model to predict car prices based on mileage.
- Evaluate the model and interpret the results.
- Provide recommendations based on the analysis.

### Starting Point

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('data/Car_Data.csv')

# Display the first few rows of the data
print(data.head())

# Calculate the correlation coefficient
correlation = data['Price'].corr(data['Mileage'])
print(f"Correlation Coefficient: {correlation}")

###################################################
# Try to compute the correlation here by yourself #
# Check the slides for the formula                #
# #### TODO ####                                  #
###################################################

#correlation_from_scratch = ...

print(f"Correlation Coefficient (from scratch): {correlation_from_scratch} Difference: {correlation - correlation_from_scratch}")

# Now that we know there is a correlation between Price and Mileage, let's build a regression model

# Build a simple linear regression model
X = data['Mileage'].values.reshape(-1, 1)
y = data['Price'].values

###################################
# Build the regression model here #
# #### TODO ####                  #
###################################

# We can do it from scratch

# Algorithm:
# f(X) = a*X + b
# 1. Calculate the means of X and y
# mean_X = ...
# mean_y = ...
# 2. Calculate the slope B1 = sum((X - mean_X) * (y - mean_y)) / sum((X - mean_X)^2)
# slope = ...
# 3. Calculate the intercept B0 = mean_y - a * mean_X
# intercept = ...
# 4. Predict the prices using the model: predicted_prices = slope * X + intercept

class LinearRegressionFromScratch:
    def __init__(self):
        self.slope = None
        self.intercept = None

    def fit(self, X, y):
        # mean_X = ...
        # mean_y = ...
        # self.slope = ...
        # self.intercept = ...
        # #### TODO ####

    def predict(self, X):
        # return ...
        # #### TODO ####

model = LinearRegressionFromScratch()
model.fit(X, y)
predicted_prices = model.predict(X)

# Calculate the mean squared error
mse = mean_squared_error(y, predicted_prices)
print(f"Mean Squared Error: {mse}")

# Let's plot the data and the regression line

# We can use matplotlib to plot the data and the regression line
# Check the documentation for matplotlib here: https://matplotlib.org/stable/gallery/index.html

plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Car Price Prediction using Linear Regression from Scratch")
plt.legend()
plt.show()

# Or we can use scikit-learn's LinearRegression model
# Look up the documentation for LinearRegression in scikit-learn
# Here: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

# model = ...
# model.fit(X, y)
# predicted_prices = ...
# #### TODO ####

###################################
# Evaluate the model here         #
# #### TODO ####                  #
###################################

# Calculate the mean squared error
mse = mean_squared_error(y, predicted_prices)
print(f"Mean Squared Error: {mse}")

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Car Price Prediction using Linear Regression (scikit-learn)")
plt.legend()
plt.savefig('car_price_prediction.png')
plt.show()
```

## Exercise 4: AB testing (Hypothesis Testing)

### Story

You work for an e-commerce company, and your task is to analyze the impact of a new website design on user engagement. You have collected data on user interactions with the old and new designs and want to determine if the new design leads to higher engagement.

### Tasks

- Perform hypothesis testing to determine if the new design leads to higher user engagement.
- Calculate the p-value and interpret the results.
- Provide recommendations based on the analysis.

### Starting Point

```python
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Load the data
data = pd.read_csv('data/A_B_Test_Data.csv')

# Display the first few rows of the data
print(data.head())

# Perform hypothesis testing
old_design = data[data['Design'] == 'Old']['Engagement']
new_design = data[data['Design'] == 'New']['Engagement']

def from_scratch_ttest_ind(sample1, sample2):
    #####################################
    # Implement the t-test from scratch #
    # #### TODO ####                    #
    #####################################
    # Algorithm:
    # 1. Calculate the means of the two samples
    # mean1 = ...
    # mean2 = ...
    # 2. Calculate the standard deviations of the two samples
    # std1 = ...
    # std2 = ...
    # 3. Calculate the standard errors of the two samples
    # se1 = ...
    # se2 = ...
    # 4. Calculate the t-statistic = (mean1 - mean2) / sqrt(se1^2 + se2^2)
    # t_stat = ...
    # 5. Calculate the degrees of freedom = n1 + n2 - 2
    # n1 = len(sample1)
    # n2 = len(sample2)
    # df = ...
    # 6. Calculate the p-value = 2 * (1 - cdf(abs(t-statistic), df))
    # p_value = ...
    # 7. Return the t-statistic and p-value
    return t_stat, p_value

t_stat, p_value = from_scratch_ttest_ind(old_design, new_design)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")
print("From Scratch:")
if p_value < 0.05:
    print("The new design leads to significantly higher user engagement.")
else:
    print("There is no significant difference in user engagement between the old and new designs.")

# We can use the built-in t-test function for comparison
t_stat, p_value = ttest_ind(old_design, new_design)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")
print("Using ttest_ind:")
if p_value < 0.05:
    print("The new design leads to significantly higher user engagement.")
else:
    print("There is no significant difference in user engagement between the old and new designs.")
```
