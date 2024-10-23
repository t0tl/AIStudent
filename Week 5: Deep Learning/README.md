# Task 1: Linear regression and autograd

## Story:
You want to use a single layer neural network to predict y given x. Instead of using the normal equations, you will instead use gradient descent to find the optimal weights and bias for the model. However, you need to implement the automatic differentiation yourself.

## Tasks:
- Implement the forward pass for a single layer neural network.
- Implement the backward pass for a single layer neural network.
- Implement the gradient descent algorithm to find the optimal weights and bias for the model.

## Starter code:
```python
import numpy as np

# Generate some data
np.random.seed(0)
x = np.random.rand(100, 1)
y = 2 + 3 * x + np.random.rand(100, 1)

# Initialize weights and bias
w = np.random.rand(1, 1)
b = np.random.rand(1)

# Your code here
...
```


# Task 2: XOR problem

# Task 3: MNIST and Convolutional Neural Networks

# Task 4: Recurrent Neural Networks

# Task 5: Attention