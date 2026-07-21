import numpy as np


# Dataset

X = np.array([
    [1,1],
    [1,2],
    [1,3],
    [1,4],
    [1,5]
])


Y = np.array([
    1.2,
    1.8,
    2.6,
    3.2,
    3.8
])


# Linear Regression Formula
# theta = (X.T X)^-1 X.T Y


theta = np.linalg.inv(
    X.T @ X
) @ X.T @ Y


print("Linear Regression Coefficients:")
print(theta)
lambda_value = 1


I = np.eye(X.shape[1])


theta1 = np.linalg.inv(
    X.T @ X + lambda_value * I
) @ X.T @ Y


print(theta1)

