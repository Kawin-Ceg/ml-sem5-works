import numpy as np
X = np.array([
    [1,2,60],
    [1,3,65],
    [1,4,70],
    [1,5,75],
    [1,6,80],
    [1,7,85]
],dtype=float)

Y = np.array([
    [40],
    [48],
    [55],
    [63],
    [70],
    [78]
],dtype=float)

theta = np.linalg.inv(X.T @ X) @ X.T @ Y

print("Theta :", theta)

