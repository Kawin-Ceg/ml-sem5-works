import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


X = np.array([
    [1,1],
    [1,2],
    [1,3],
    [1,4],
    [1,5]
])


Y = np.array([
    1.2,
    1.8,
    2.6,
    3.2,
    3.8
])




linear_model = LinearRegression(
    fit_intercept=False
)


linear_model.fit(X,Y)



print("Linear Regression Coefficients:")

print(linear_model.coef_)




ridge_model = Ridge(
    alpha=1,
    fit_intercept=False
)


ridge_model.fit(X,Y)



print("\nRidge Regression Coefficients:")

print(ridge_model.coef_)