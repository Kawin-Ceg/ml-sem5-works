import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Dataset

X = np.array([
    [1],
    [3],
    [5],
    [7],
    [10]
])


y = np.array([
    0,
    0,
    0,
    1,
    1
])



# Create model

model = LogisticRegression()



# Train model

model.fit(X,y)



# Parameters

print("Weight:")
print(model.coef_)


print("\nBias:")
print(model.intercept_)



# Prediction

predictions = model.predict(X)



print("\nPredictions:")
for weeks,pred in zip(X,predictions):

    print(
        weeks[0],
        "weeks ->",
        pred
    )



# Accuracy

accuracy = accuracy_score(y,predictions)

print("\nAccuracy:",accuracy)