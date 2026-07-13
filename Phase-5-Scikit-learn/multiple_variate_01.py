import numpy as np
X = np.array([
    [1,1000,2,15],
    [1,1200,2,12],
    [1,1500,3,8],
    [1,1800,3,5],
    [1,2000,4,2]
],dtype=float)

Y = np.array([
    [40],
    [45],
    [55],
    [65],
    [75]
],dtype=float)

theta = np.linalg.inv(X.T @ X) @ X.T @ Y

print("Theta :", theta)

new_house = np.array([[1,1700,3,6]])
prediction = new_house @ theta
print("Predicted Price :", prediction[0][0])