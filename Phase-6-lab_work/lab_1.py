import numpy as np

# Dataset
X = np.array([1,2,3,4,5,6])
Y = np.array([3,5,7,9,11,13])


# Add bias column (x0 = 1)
X_matrix = np.column_stack((np.ones(len(X)), X))


# Matrix method:
# theta = (X.T X)^-1 X.T Y

theta = np.linalg.inv(X_matrix.T @ X_matrix) @ X_matrix.T @ Y


# Extract coefficients
b = theta[0]
w = theta[1]

print("Intercept:", b)
print("Slope:", w)


# Prediction
Y_pred = X_matrix @ theta


print("\nPredictions:")
print(Y_pred)


# Evaluation Metrics

mae = np.mean(np.abs(Y - Y_pred))

mse = np.mean((Y - Y_pred)**2)

rmse = np.sqrt(mse)


ss_total = np.sum((Y - np.mean(Y))**2)

ss_residual = np.sum((Y - Y_pred)**2)

r2 = 1 - (ss_residual / ss_total)



print("\nMAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2 Score:", r2)