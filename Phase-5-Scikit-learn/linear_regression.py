import numpy as np; 

x = np.array([1000,1200,1500,1800,2000],dtype=float)
y = np.array([30,35,45,50,60],dtype=float)

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


print("Coefficient a:", a)
print("Coefficient b:", b)

# new prediction for x = 1700
x_new = 1700

prediction = a*x_new + b
print("Prediction for x = 1700:", prediction)