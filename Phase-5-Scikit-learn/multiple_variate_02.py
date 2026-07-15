import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data

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

model = LinearRegression()
model.fit(X,Y)

# value of a is the coefficient of the model
print("Coefficients:", model.coef_)

# value of b is the intercept of the model
print("Intercept:", model.intercept_)

answer = model.predict([[1,8,90]])
print("Marks Obtained : ",answer)