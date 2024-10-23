# Exercise 1: Analyzing (Descriptive Statistics + Data Visualization)


# Exercise 2: Probability (Probability Theory + Simulation)


# Exercise 3:  


# Exercise 4: AB testing (Hypothesis Testing)


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