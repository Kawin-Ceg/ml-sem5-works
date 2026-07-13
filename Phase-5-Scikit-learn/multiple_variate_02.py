import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([
    [1000, 2, 15],
    [1200, 2, 12],
    [1500, 3, 8],
    [1800, 3, 5],
    [2000, 4, 2]
])

Y = np.array([40, 45, 55, 65, 75])

model = LinearRegression()
model.fit(X,Y)

# value of a is the coefficient of the model
print("Coefficients:", model.coef_)

# value of b is the intercept of the model
print("Intercept:", model.intercept_)

new_house = np.array([[1700,3,6]])

prediction = model.predict(new_house)
print("Predicted Price for new house:", prediction[0])