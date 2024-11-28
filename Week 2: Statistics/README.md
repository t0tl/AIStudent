# Installing Required Packages with `uv`
For this week we will work with external packages like `numpy`, `scipy`, `pandas`, `matplotlib`, `statsmodels` and `sklearn`. To do this, we recommend you to use `uv` for managing your virtual environment. You can install it by running `pip install uv`. Thereafter, you can just do `uv run <name_of_file.py>` to run your code if there is an inline dependency like

```python
# /// script
# dependencies = [
#   "numpy==2.1.3",
# ]
# ///
import numpy as np

print(np.__version__)
```

Uv is an alternative to `pip` and `venv`. It allows for inline dependencies, faster installs, and better management of virtual environments using lockfiles. You can find more information about `uv` [here](https://docs.astral.sh/uv/).

# Exercise 1: Analyzing (Descriptive Statistics + Data Visualization)

## Contents:
Mean, Median, Mode, Variance, Standard Deviation, Range, Skewness, Kursosis, Histograms, Box Plots, Scatter Plots, Bar charts

## Story:
You have been given a dataset containing information about students' test scores. Your task is to analyze the data and provide insights to help improve student performance.

## Tasks:
- Load the dataset
- Compute descriptive statistics (mean, median, mode, variance, standard deviation, range, skewness, kurtosis)
- Visualize the data using histograms, box plots, scatter plots, and bar charts
- Explain the insights you gained from the analysis

# Exercise 2: Probability (Probability Theory + Simulation)

## Topics:
Probability, Conditional Probability, Bayes' Theorem, Random Variables, Probability Distributions, Simulation

## Tasks:
- Simulate 1000 coin tosses, calculate the probability of getting heads, and compare it with the theoretical probability.
- Simulate 1000 dice rolls, calculate the probability of getting a prime number, and compare it with the theoretical probability. Calculate the conditional probability of getting a prime number given that the number is odd.
- Use Monte Carlo simulation to estimate the value of $\pi$ by generating random points in a unit square and calculating the ratio of points inside a quarter circle to the total points.

# Exercise 3: Correlation and Regression (Correlation + Regression Analysis)

## Story:
You work for a car dealership, and your task is to analyze the relationship between car prices and their mileage. You want to determine if there is a correlation between the two variables and build a regression model to predict car prices based on mileage.

## Tasks:
- Calculate the correlation coefficient between car prices and mileage.
- Build a simple linear regression model to predict car prices based on mileage.
- Evaluate the model and interpret the results.
- Provide recommendations based on the analysis.

## Starting Point:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv('data/car_data.csv')

# Display the first few rows of the data
print(data.head())

# Calculate the correlation coefficient
correlation = data['Price'].corr(data['Mileage'])
print(f"Correlation Coefficient: {correlation}")

# Build a simple linear regression model
X = data['Mileage'].values.reshape(-1, 1)
y = data['Price'].values

###################################
# Build the regression model here #
###################################

# model = ...
# model.fit(X, y)
# predicted_prices = ...

###################################
# Evaluate the model here         #
###################################

# mse = ...

# Plot the data and the regression line
plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Car Price Prediction")
plt.legend()
plt.show()
```

# Exercise 4: AB testing (Hypothesis Testing)

## Story:
You work for an e-commerce company, and your task is to analyze the impact of a new website design on user engagement. You have collected data on user interactions with the old and new designs and want to determine if the new design leads to higher engagement.

## Tasks:
- Perform hypothesis testing to determine if the new design leads to higher user engagement.
- Calculate the p-value and interpret the results.
- Provide recommendations based on the analysis.

## Starting Point:
```python
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

# Load the data
data = pd.read_csv('data/ab_test_data.csv')

# Display the first few rows of the data
print(data.head())

# Perform hypothesis testing
old_design = data[data['Design'] == 'Old']['Engagement']
new_design = data[data['Design'] == 'New']['Engagement']

def from_scratch_ttest_ind(sample1, sample2):
    #####################################
    # Implement the t-test from scratch #
    #####################################
    return t_stat, p_value

t_stat, p_value = from_scratch_ttest_ind(old_design, new_design)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")


# Interpret the results
if p_value < 0.05:
    print("The new design leads to significantly higher user engagement.")
else:
    print("There is no significant difference in user engagement between the old and new designs.")

# We can use the built-in t-test function for comparison
t_stat, p_value = ttest_ind(old_design, new_design)
print(f"t-statistic: {t_stat}")
print(f"p-value: {p_value}")
if p_value < 0.05:
    print("The new design leads to significantly higher user engagement.")
else:
    print("There is no significant difference in user engagement between the old and new designs.")
```


# Exercise 5: Predicting Housing Prices (Linear Regression + Gauss-Markov Assumptions)

## Story:
You work for a real estate agency, and your task is to predict house prices based on various features like the number of bedrooms, square footage, and distance to the city center. You’ll need to build a linear regression model and evaluate its assumptions based on the Gauss-Markov theorem.

## Tasks:
- Build a simple linear regression model to predict house prices based on square footage.
- Explain and evaluate if the Gauss-Markov assumptions for the linear regression model.
    * Linearity
    * Independence
    * Homoscedasticity (constant variance of errors)
    * No multicollinearity (for multivariate regression)
    * Normal distribution of errors (optional for this exercise)
- What are the implications if a specific assumption is violated?

## Starting Point:
```python
# TODO: Add uv requirements here
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

# Sample data: square footage and corresponding house prices
square_footage = np.array([1500, 1800, 2400, 3000, 3500, 4000]).reshape(-1, 1)
house_prices = np.array([300000, 360000, 480000, 600000, 700000, 800000])

# Fit a simple linear regression model
model = LinearRegression()
model.fit(square_footage, house_prices)

# Predict prices
predicted_prices = model.predict(square_footage)

# Evaluate the model
mse = mean_squared_error(house_prices, predicted_prices)
print(f"Mean Squared Error: {mse}")
print(f"Model Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Plot actual vs predicted prices
plt.scatter(square_footage, house_prices, color='blue', label='Actual Prices')
plt.plot(square_footage, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel("Square Footage")
plt.ylabel("House Prices")
plt.title("House Price Prediction")
plt.legend()
plt.show()

# Check residuals for homoscedasticity using statsmodels
X = sm.add_constant(square_footage)  # Add constant for intercept
ols_model = sm.OLS(house_prices, X).fit()
print(ols_model.summary())
```