import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1000],
    [1200],
    [1500],
    [1800],
    [2000]
])

Y = np.array([30,35,45,55,60])

model = LinearRegression()
model.fit(X,Y)

# value of a is the coefficient of the model
print("Coefficients:", model.coef_)

# value of b is the intercept of the model
print("Intercept:", model.intercept_)

new_house = np.array([[1700]])

prediction = model.predict(new_house)
print("Prediction for new house:", prediction)