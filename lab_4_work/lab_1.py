import numpy as np

# Dataset
X = np.array([1,3,5,7,10], dtype=float)
y = np.array([0,0,0,1,1], dtype=float)

# reshape X
X = X.reshape(-1,1)


# Initialize parameters
w = np.zeros((1,1))
b = 0


# Sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))


# Training parameters
learning_rate = 0.01
epochs = 10000


# Gradient Descent
for i in range(epochs):

    # Forward propagation
    z = np.dot(X,w) + b

    predictions = sigmoid(z)


    # Calculate gradients

    dw = np.dot(X.T,(predictions-y.reshape(-1,1))) / len(y)

    db = np.sum(predictions-y.reshape(-1,1)) / len(y)


    # Update weights

    w = w - learning_rate*dw

    b = b - learning_rate*db



print("Weight:",w)
print("Bias:",b)



# Prediction function

def predict(X):

    z = np.dot(X,w)+b

    probability = sigmoid(z)

    return (probability >=0.5).astype(int)



# Testing

test_data = np.array([[1],[3],[5],[7],[10]])

result = predict(test_data)


print("\nPredictions:")
for weeks,pred in zip(test_data,result):
    print(
        weeks[0],
        "weeks ->",
        pred[0]
    )