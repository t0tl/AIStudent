# /// script
# dependencies = [
#   "numpy==2.1.3",
# ]
# ///

import numpy as np

# Setting the seed
np.random.seed(0)

# Coin flip simulation
n = 1000
p = 0.5
coin_flips = np.random.randint(0, 2, n) # 0: tails, 1: heads

# Counting the number of heads
heads = np.sum(coin_flips)

# Calculating the probability of heads
p_heads = heads / n


# Dice roll simulation
n = 1000
dice_rolls = np.random.randint(1, 7, n)

# Counting the number of prime numbers
prime_rolls = np.sum((dice_rolls == 2) | (dice_rolls == 3) | (dice_rolls == 5))
print("In theoretical probability, the probability of rolling a prime number is 3/6 = 0.5")
print("The probability of rolling a prime number is", prime_rolls / n)

# Counting the number of odd rolls
odd_rolls = np.sum(dice_rolls % 2 != 0)
print("In theoretical probability, the probability of rolling an odd number is 3/6 = 0.5")
print("The probability of rolling an odd number is", odd_rolls / n)

# Counting P(roll a prime number | roll an odd number) = P(roll a prime number and roll an odd number) / P(roll an odd number)

# Counting the number of prime and odd rolls
prime_and_odd_rolls = np.sum(((dice_rolls == 2) | (dice_rolls == 3) | (dice_rolls == 5)) & (dice_rolls % 2 != 0))

# Calculating the probability of rolling a prime number given that the roll is odd
p_prime_given_odd = prime_and_odd_rolls / odd_rolls
print("In theoretical probability, the probability of rolling a prime number given that the roll is odd is 2/3 = 0.6666666666666666")
print("The probability of rolling a prime number given that the roll is odd is", p_prime_given_odd)

# Monte Carlo simulation to estimate pi
n = 100000
x = np.random.uniform(-1, 1, n)
y = np.random.uniform(-1, 1, n)

# Counting the number of points inside the circle
inside_circle = np.sum(x**2 + y**2 <= 1)

# Calculating the probability of a point being inside the circle
p_inside_circle = inside_circle / n

# Estimating pi
pi_estimate = 4 * p_inside_circle
print("In theoretical probability, the value of pi is 3.141592653589793")
print("The estimated value of pi is", pi_estimate)