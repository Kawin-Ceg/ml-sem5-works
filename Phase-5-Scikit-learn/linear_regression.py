import numpy as np; 

x = np.array([1,2,3,4,5,6],dtype=float)
y = np.array([3,5,7,9,11,13],dtype=float)

# calculated the mean of x and y 
x_mean = np.mean(x)
y_mean = np.mean(y) 

print("Mean of x:", x_mean)
print("Mean of y:", y_mean)

# calculate the mean of x*y and x^2
xy_mean = np.mean(x*y)
x2_mean = np.mean(x**2)

# calculate the coefficients a and b for the linear regression line y = ax + b
a = (xy_mean - x_mean*y_mean) / (x2_mean - x_mean**2)
b = y_mean - a*x_mean


print("Coefficient a:", round(a,4))
print("Coefficient b:", round(b,4))


x_new = 7
y_new = 10 

prediction_1 = a*x_new + b
prediction_2 = a*y_new + b
print("Prediction for x = 7:", round(prediction_1,4))
print("Prediction for x = 10: ",round(prediction_2,4))