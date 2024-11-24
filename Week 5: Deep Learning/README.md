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

# Task 2: MNIST and Multilayer Perceptron

```python
import numpy as np

# Generate toy data
np.random.seed(42)
x = np.random.rand(100, 1)  # 100 samples, 1 feature
y = (x > 0.5).astype(float)  # Binary target (0 or 1)

# Initialize parameters
input_dim = 1
hidden_dim = 5  # Number of neurons in the hidden layer
output_dim = 1

# Initialize weights and biases
W1 = np.random.randn(input_dim, hidden_dim) * 0.1  # Input to hidden layer weights
b1 = np.zeros((1, hidden_dim))  # Hidden layer biases
W2 = np.random.randn(hidden_dim, output_dim) * 0.1  # Hidden to output weights
b2 = np.zeros((1, output_dim))  # Output layer biases

# Sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Derivative of sigmoid
def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))

# Forward pass
def forward(x, W1, b1, W2, b2):
    z1 = np.dot(x, W1) + b1  # Linear transformation for hidden layer
    a1 = sigmoid(z1)         # Activation for hidden layer
    z2 = np.dot(a1, W2) + b2 # Linear transformation for output layer
    a2 = sigmoid(z2)         # Activation for output layer
    return z1, a1, z2, a2

# Backward pass
def backward(x, y, z1, a1, z2, a2, W1, W2):
    m = x.shape[0]  # Number of samples
    dz2 = a2 - y  # Output error
    dW2 = np.dot(a1.T, dz2) / m  # Gradient for W2
    db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Gradient for b2
    
    da1 = np.dot(dz2, W2.T)  # Backpropagate error to hidden layer
    dz1 = da1 * sigmoid_derivative(z1)  # Apply activation function derivative
    dW1 = np.dot(x.T, dz1) / m  # Gradient for W1
    db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient for b1
    
    return dW1, db1, dW2, db2

# Gradient descent update
def update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    return W1, b1, W2, b2

# Loss function: Binary Cross-Entropy
def compute_loss(y, y_hat):
    m = y.shape[0]
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
    return loss

# Training loop
learning_rate = 0.1
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    z1, a1, z2, a2 = forward(x, W1, b1, W2, b2)
    
    # Compute loss
    loss = compute_loss(y, a2)
    losses.append(loss)
    
    # Backward pass
    dW1, db1, dW2, db2 = backward(x, y, z1, a1, z2, a2, W1, W2)
    
    # Update parameters
    W1, b1, W2, b2 = update_parameters(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
    
    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.4f}")

# Visualize the loss
import matplotlib.pyplot as plt

plt.plot(range(num_epochs), losses)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
```


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Define MLP Model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        ### Implement here

    def forward(self, x):
        x = x.view(-1, 28 * 28)           # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize Model, Loss, and Optimizer
model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
```



# Task 3: MNIST and Convolutional Neural Networks

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        ### Implement here


    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)  # Flatten the tensor
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize Model, Loss, and Optimizer
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total:.2f}%")
```