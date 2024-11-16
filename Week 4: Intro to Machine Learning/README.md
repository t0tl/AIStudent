# Introduction to Machine Learning

We finally arrived to one of the most importants parts of this couse, *Machine Learning*. You probably have heard about it, but what is it? Why is it so famous? Why everyone is talking about it? Well, let's find out!

## What is Machine Learning?

Machine Learning is a subfield of Artificial Intelligence (AI) that provides systems the ability to automatically learn and improve from experience without being explicitly programmed. We use maths and statistics to make predictions or decisions based on data. In other words, we use data to make decisions.

![](/Week%204:%20Intro%20to%20Machine%20Learning/ML.png)

There are 3 types of Machine Learning:

1. **Supervised Learning**: The algorithm learns from labeled data, and makes predictions based on that data. For example, we can use a dataset of houses with their prices to predict the price of a new house.

The goal is to train a model $f$ that maps inputs $X$ to outputs $y$, such that

$$y = f(X)$$

Inside Supervised Learning, we have 2 types of problems:
    - **Classification**: The output variable is a category, such as "spam" or "not spam". Here $y\in\{c_1, c_2, ..., c_n\}$, where $c_i$ is a category, for the spam example $y\in\{"spam", "not spam"\}$.
    - **Regression**: The output variable is a real value, such as "price" or "weight". Here $y\in\mathbb{R}$.

2. **Unsupervised Learning**: The algorithm learns from unlabeled data, and tries to find patterns in the data. For example, we can use a dataset of houses without prices to find patterns in the data.

3. **Reinforcement Learning**: The algorithm learns by interacting with its environment. It receives rewards for performing correctly and penalties for performing incorrectly. For example, we can use a dataset of a robot that learns to walk.

In this exercise, we will focus on the first 2 types of Machine Learning. Let's get through it!

## Tasks

### Task 1: Data Preprocessing

As we mentioned before, Machine Learning is all about models learning from data to make predictions or decisions. But before we can train a model, we need to prepare the data. This process is called Data Preprocessing. In this task, you will learn how to handle the data before training a model.

First let's load the data, we will use the Heart Disease dataset from the UCI Machine Learning Repository. This dataset contains 303 samples with 13 features each. The goal is to predict whether a patient has heart disease or not. You can find the dataset on Kaggle [here](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

Your mission is to load the data, preprocess it, and split it into training and test sets. 

Tasks:

1. Load the data using the following code:

```python
import pandas as pd

data = pd.read_csv("heart.csv")
```

2. Identify the features and the target variable. The target variable is the variable we want to predict, in this case, it is the `target` column.

3. Identify missing values in the dataset, decide if to remove or fill them.

4. Handle categorical variables. In this dataset, we have a categorical variable called `cp` that represents the chest pain type. 

5. Split the data into training and test sets. You can use the following code:

```python
from sklearn.model_selection import train_test_split

X = data.drop("target", axis=1)
y = data["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Note that we split the data into training and test sets. The training set is used to train the model, while the test set is used to evaluate the model. We use 80% of the data for training and 20% for testing.

Some additional tasks you can do:

1. Normalize the data. Normalizing the data is important because it can help the model converge faster. 

2. Create new features. Sometimes creating new features can help the model learn better. For example, you can create a new feature by combining two features.


### Task 2: Training a ML Algorithm for Classification

Now that we have our data ready, we can start training a Machine Learning model. As you noticed in the first task our data is labeled, which means we can use a Supervised Learning algorithm. In this task, you will learn how to train a model for classification.

We will first train a simple Perceptron model, and then move on to a more advanced models, your first task is to study the following code and understand how it works:

```python
import numpy as np
class Perceptron:
    """Perceptron classifier.
    Parameters
    ------------
    eta : float
    Learning rate (between 0.0 and 1.0)
    n_iter : int
    Passes over the training dataset.
    random_state : int
    Random number generator seed for random weight
    initialization.
    Attributes
    -----------
    w_ : 1d-array
    Weights after fitting.
    b_ : Scalar
    Bias unit after fitting.
    errors_ : list
    Number of misclassifications (updates) in each epoch.
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
        Training vectors, where n_samples is the number of samples and n_features is the number of features.
        y : array-like, shape = [n_samples]
        Target values.

        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
        self.b_ = 0
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
```

As you can see we have a class called `Perceptron` that has 3 main methods:

- `fit`: This method is used to train the model. It receives the input data `X` and the target values `y`, and updates the weights of the model.
- `net_input`: This method calculates the net input of the model.
- `predict`: This method predicts the class label after the unit step.

Now, your task is to train the Perceptron model using the previously loaded data. You can use the following code to train the model:

```python

model = Perceptron(eta=0.1, n_iter=10)
model.fit(X_train, y_train)
```

Now that we have trained our model we can evaluate it using the test data. You can use the following code to evaluate the model:

```python
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
```

We can visualize the model's decision boundaries using the following code:

```python
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                       np.arange(x2_min, x2_max, 0.01))
Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=ListedColormap(('red', 'blue')))
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.show()
```

Although the Perceptron model is a simple model, it can be very useful for linearly separable data. In the next tasks, we will train more advanced models.

Scikit-learn is a powerful library that provides simple and efficient tools for data mining and data analysis. It is built on NumPy, SciPy, and Matplotlib. In this task, you will learn how to use Scikit-learn to train a more advanced model for classification.

Your task is to test the following models finding the best one for the dataset. You will use the following models described in the Scikit-learn documentation:

- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) 
- [Support Vector Machine](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
- [Decision Tree](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
- [Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)

You can use the following code to train and evaluate the models, you will notice that the API is the same as the Perceptron model:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

model_lr = LogisticRegression()
model_svm = SVC()
model_dt = DecisionTreeClassifier()
model_rf = RandomForestClassifier()

# Train the models
# ...

# Evaluate the models
# ...
```

### Task 3: Best Practices for Model Evaluation and Hyperparameter Tuning

In this task, you will learn some best practices for model evaluation and hyperparameter tuning. 

When training a Machine Learning model, it is important to evaluate the model properly. There are several metrics that can be used to evaluate a model, such as accuracy, precision, recall, F1-score, and ROC-AUC.

In this task, you will learn how to evaluate a model using the following metrics:

- Accuracy: The proportion of correctly classified instances.
- Precision: The proportion of correctly classified positive instances among all instances classified as positive.
- Recall: The proportion of correctly classified positive instances among all actual positive instances.
- F1-score: The harmonic mean of precision and recall.
- ROC-AUC: The area under the receiver operating characteristic curve.

You can use the following code to calculate these metrics:

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")
print(f"ROC-AUC: {roc_auc}")
```

Another important aspect of training a Machine Learning model is hyperparameter tuning. Hyperparameters are parameters that are set before the learning process begins. They control the learning process and affect the performance of the model.

In this task, you will learn how to tune the hyperparameters of a model using Grid Search. Grid Search is a technique that searches for the best hyperparameters by evaluating all possible combinations of hyperparameters.

You can use the following code to perform Grid Search:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score}")
```

### Task 4: Training a ML Algorithm for Regression

For this task, you will train a Machine Learning model for regression. Regression is a type of supervised learning that predicts a continuous value. In this task, you will use the Boston Housing dataset from the UCI Machine Learning Repository. This dataset contains 506 samples with 13 features each. The goal is to predict the median value of owner-occupied homes.

Your task is to load the data, preprocess it, and split it into training and test sets. You can use the following code to load the data:

```python
data = pd.read_csv("boston.csv")
```

You can use the following code to split the data into training and test sets:

```python
X = data.drop("MEDV", axis=1)
y = data["MEDV"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Now that you have the data ready, you can train a Machine Learning model for regression. You can use the following models described in the Scikit-learn documentation:

- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Support Vector Machine for Regression](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html)
- [Decision Tree for Regression](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [Random Forest for Regression](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

You can use the following code to train and evaluate the models:

```python
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

model_lr = LinearRegression()
model_svm = SVR()
model_dt = DecisionTreeRegressor()
model_rf = RandomForestRegressor()

# Train the models
# ...

# Evaluate the models
# ...
```

You can use the following code to calculate the mean squared error (MSE) and the coefficient of determination (R^2) for the regression models:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

Where $y_i$ is the true value and $\hat{y}_i$ is the predicted value.

```python
from sklearn.metrics import mean_squared_error, r2_score

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse}")
print(f"R^2: {r2}")
```

You can also visualize the predicted values against the true values using the following code:

```python
plt.scatter(y_test, y_pred)
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.show()
```

### Task 5: Working with Unlabeled Data - Clustering

In this task, you will learn how to work with unlabeled data using clustering. Clustering is a type of unsupervised learning that groups similar data points together. In this task, you will use the Iris dataset from the UCI Machine Learning Repository. This dataset contains 150 samples with 4 features each. The goal is to group the samples into clusters based on their features.

Your task is to load the data, preprocess it, and train a clustering model. You can use the following code to load the data:

```python
data = pd.read_csv("iris.csv")
```

You can use the following code to preprocess the data:

```python
X = data.drop("species", axis=1)
```

Now that you have the data ready, you can train a clustering model. You can use the following models described in the Scikit-learn documentation:

- [KMeans](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
- [Agglomerative Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
- [Gaussian Mixture Model](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html)

You can use the following code to train and evaluate the models:

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, GaussianMixture

model_kmeans = KMeans(n_clusters=3)
model_dbscan = DBSCAN(eps=0.5, min_samples=5)
model_agg = AgglomerativeClustering(n_clusters=3)
model_gmm = GaussianMixture(n_components=3)

# Train the models
# ...

# Evaluate the models
# ...
```

You can use the following code to visualize the clusters:

```python
plt.scatter(X["sepal_length"], X["sepal_width"], c=y_pred, cmap='viridis')
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.show()
```