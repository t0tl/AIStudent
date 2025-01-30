import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import mode, t
from sklearn.linear_model import LinearRegression

# Load datasets
student_scores = pd.read_csv('data/Student_Scores_Data.csv')
car_data = pd.read_csv('data/Car_Data.csv')
ab_test_data = pd.read_csv('data/A_B_Test_Data.csv')

# Ensure correct column names
print("Column names:")
print(student_scores.columns)
print(car_data.columns)
print(ab_test_data.columns)

# Display the first few rows of the datasets
print("First few rows:")
print(student_scores.head())
print(car_data.head())
print(ab_test_data.head())

# Exercise 1: Descriptive Statistics & Visualization
mean_score = np.mean(student_scores['Score'])
median_score = np.median(student_scores['Score'])
mode_score = mode(student_scores['Score'])
variance_score = np.var(student_scores['Score'], ddof=1)
std_dev_score = np.std(student_scores['Score'], ddof=1)

print(f"Mean: {mean_score}, Median: {median_score}, Mode: {mode_score}, Variance: {variance_score}, Standard Deviation: {std_dev_score}")

plt.hist(student_scores['Score'], bins=10, edgecolor='black')
plt.title('Histogram of Student Scores')
plt.xlabel('Score')
plt.ylabel('Frequency')
plt.show()

plt.boxplot(student_scores['Score'])
plt.title('Box Plot of Student Scores')
plt.show()

# Exercise 2: Probability Simulations
coin_tosses = np.random.choice(['H', 'T'], size=1000)
prob_heads = np.mean(coin_tosses == 'H')
print(f"Probability of getting heads: {prob_heads}")

dice_rolls = np.random.randint(1, 7, size=1000)
primes = {2, 3, 5}
prob_prime = np.mean(np.isin(dice_rolls, list(primes)))
prob_prime_given_odd = np.mean(np.isin(dice_rolls[dice_rolls % 2 == 1], list(primes)))
print(f"Probability of prime: {prob_prime}")
print(f"Conditional probability of prime given odd: {prob_prime_given_odd}")

# Monte Carlo Estimation of Pi
n = 1000000
points = np.random.rand(n, 2)
in_circle = np.sum(points[:, 0]**2 + points[:, 1]**2 <= 1)
pi_estimate = 4 * in_circle / n
print(f"Estimated Pi: {pi_estimate}")

# Exercise 3: Correlation and Regression
correlation = np.corrcoef(car_data['Mileage'], car_data['Price'])[0, 1]
print(f"Correlation Coefficient: {correlation}")



# Manual calculation of correlation coefficient
X = car_data['Mileage'].values.reshape(-1, 1)
y = car_data['Price'].values.reshape(-1, 1)
n = len(X)
X_mean = np.mean(X)
y_mean = np.mean(y)
numerator = np.sum((X - X_mean) * (y - y_mean))
denominator = np.sqrt(np.sum((X - X_mean)**2) * np.sum((y - y_mean)**2))
correlation_manual = numerator / denominator
print(f"Correlation Coefficient (Manual): {correlation_manual}")

if np.isclose(correlation, correlation_manual):
    print("Correlation coefficients match.")
else:
    print("Correlation coefficients do not match.")

# Linear Regression
n = len(y)
X_mean = np.mean(X)
y_mean = np.mean(y)
B1 = np.sum((X - X_mean) * (y - y_mean)) / np.sum((X - X_mean)**2)
B0 = y_mean - B1 * X_mean
predicted_prices = B0 + B1 * X.flatten()
mse = np.mean((y - predicted_prices) ** 2)
print(f"Mean Squared Error: {mse}")

plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices, color='red', label='Predicted Prices')
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Car Price Prediction")
plt.legend()
plt.show()

# We can do this with sklearn as well
regressor = LinearRegression()
regressor.fit(X, y)
predicted_prices_sklearn = regressor.predict(X)
mse_sklearn = np.mean((y - predicted_prices_sklearn) ** 2)
print(f"Mean Squared Error (Sklearn): {mse_sklearn}")

plt.scatter(X, y, color='blue', label='Actual Prices')
plt.plot(X, predicted_prices_sklearn, color='red', label='Predicted Prices')
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Car Price Prediction (Sklearn)")
plt.legend()
plt.show()

# Exercise 4: A/B Testing
old_design = ab_test_data[ab_test_data['Design'] == 'Old']['Engagement'].values
new_design = ab_test_data[ab_test_data['Design'] == 'New']['Engagement'].values

mean_old = np.mean(old_design)
mean_new = np.mean(new_design)
var_old = np.var(old_design, ddof=1)
var_new = np.var(new_design, ddof=1)
n_old = len(old_design)
n_new = len(new_design)

t_stat = (mean_new - mean_old) / np.sqrt(var_old/n_old + var_new/n_new)
df = (var_old/n_old + var_new/n_new)**2 / ((var_old/n_old)**2/(n_old-1) + (var_new/n_new)**2/(n_new-1))
p_value = 2 * (1 - t.cdf(np.abs(t_stat), df))

print(f"t-statistic: {t_stat}, p-value: {p_value}")
if p_value < 0.05:
    print("The new design leads to significantly higher user engagement.")
else:
    print("No significant difference in user engagement.")
