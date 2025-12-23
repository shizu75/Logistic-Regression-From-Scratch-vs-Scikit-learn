import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X_train = pd.read_csv(r"D:\Internship\train_X.csv")
Y_train = pd.read_csv(r"D:\Internship\train_Y.csv")
X_test = pd.read_csv(r"D:\Internship\test_X.csv")
Y_test = pd.read_csv(r"D:\Internship\test_Y.csv")

X_train = X_train.drop("Id", axis = 1)
Y_train = Y_train.drop("Id", axis = 1)
X_test = X_test.drop("Id", axis = 1)
Y_test = Y_test.drop("Id", axis = 1)

X_train = X_train.values
Y_train = Y_train.values
X_test = X_test.values
Y_test = Y_test.values

X_train = X_train.T
Y_train = Y_train.reshape(1, X_train.shape[1])

X_test = X_test.T
Y_test = Y_test.reshape(1, X_test.shape[1])

def model(X, Y, iterations, learning_rate):
    m = X_train.shape[1]
    n = X_train.shape[0]

    W = np.zeros((n,1))
    B = 0
    costs = []

    for i in range(iterations):
        Z = np.dot(W.T, X) + B
        A = sigmoid(Z)
        cost = -(1/m)*np.sum(Y*np.log(A) + (1 - Y)*np.log(1 - A))
        dW = (1/m)*np.dot(A - Y, X.T)
        dB = (1/m)*np.sum(A - Y)
        W = W - learning_rate * (dW.T)
        B = B - learning_rate * dB
        costs.append(cost)

        if(i%(iterations/10) == 0):
            print("Cost after", i, "iteration is: ", cost)
        
    return W, B, costs
        
        
def accuracy(X, Y, W, B):
   Z = np.dot(W.T, X) + B
   A= sigmoid(Z)
   A = np.where(A>0.5, 1, 0)
   Accuracy = (1 - np.sum(np.absolute(A - Y)/Y.shape[1]))*100
   print("Accuracy of the model is: ",  Accuracy, "%")
    
def sigmoid(X):
    return (1/(1+np.exp(-X)))

iterations = 10000
learning_rate = 0.0012
W, B, costs = model(X_train, Y_train, iterations, learning_rate)
accuracy(X_test, Y_test, W, B)
plt.figure()
plt.plot(np.arange(iterations), costs, color = 'black')
plt.show()
