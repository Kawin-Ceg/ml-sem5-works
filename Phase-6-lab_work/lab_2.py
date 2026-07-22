import numpy as np

X = np.array([
    [2,60],
    [3,65],
    [4,70],
    [5,75],
    [6,80],
    [7,85]
])

Y = np.array([
    40,
    48,
    55,
    63,
    70,
    78
])

X_matrix = np.column_stack(
    (
        np.ones(len(X)),
        X
    )
)



theta = np.linalg.inv(
    X_matrix.T @ X_matrix
) @ X_matrix.T @ Y


print("Coefficients:")
print(theta)


Y_pred = X_matrix @ theta

print("\nPredicted Marks:")
print(Y_pred)


mae = np.mean(
    np.abs(Y-Y_pred)
)

mse = np.mean(
    (Y-Y_pred)**2
)

rmse = np.sqrt(mse)
ss_total = np.sum(
    (Y-np.mean(Y))**2
)

ss_res = np.sum(
    (Y-Y_pred)**2
)

r2 = 1-(ss_res/ss_total)
print("\nMAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
print("R2 Score:",r2)