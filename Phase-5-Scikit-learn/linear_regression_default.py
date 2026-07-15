import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1],
    [2],
    [3],
    [4],
    [5],
    [6]
])

Y = np.array([3,5,7,9,11,13])

model = LinearRegression()
model.fit(X,Y)

# value of a is the coefficient of the model
print("Coefficients:", model.coef_)

# value of b is the intercept of the model
print("Intercept:", round(model.intercept_,4))

print(f"Y = {round(model.coef_[0],4)}X + {round(model.intercept_,4)}")