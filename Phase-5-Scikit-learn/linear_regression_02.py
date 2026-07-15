import numpy as np

X = np.array([[1,1],[1,2],[1,3],[1,4],[1,5],[1,6]],dtype=float)

Y = np.array([[3],[5],[7],[9],[11],[13]],dtype=float)

print("X:", X.shape)
print("Y:", Y.shape)

XTX = X.T @ X
inverse = np.linalg.inv(XTX)
XTY = X.T @ Y

theta = inverse @ XTY
print("Coefficients (theta):", theta)

