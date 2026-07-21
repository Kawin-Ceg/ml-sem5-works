import numpy as np

from sklearn.linear_model import LinearRegression

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)



# Dataset 3

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


# Create model

model = LinearRegression()


# Train

model.fit(X,Y)



print("Intercept:",
      model.intercept_)


print("Coefficients:",
      model.coef_)



# Prediction

Y_pred=model.predict(X)



print("\nPredictions:")
print(Y_pred)



# Metrics

print("\nMAE:",
      mean_absolute_error(Y,Y_pred))


print("MSE:",
      mean_squared_error(Y,Y_pred))


print("RMSE:",
      np.sqrt(mean_squared_error(Y,Y_pred)))


print("R2:",
      r2_score(Y,Y_pred))