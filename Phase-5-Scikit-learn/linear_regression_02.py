import numpy as np

X = np.array([[1,1000],[1,1200],[1,1500],[1,1800],[1,2000]],dtype=float)

Y = np.array([[30],[35],[45],[60],[55]],dtype=float)

print("X:", X.shape)
print("Y:", Y.shape)

XTX = X.T @ X
inverse = np.linalg.inv(XTX)
XTY = X.T @ Y

theta = inverse @ XTY
print("Coefficients (theta):", theta)

new_house = np.array([[1,1700]])
prediction = new_house @ theta
print("Prediction for new house:", prediction)