import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score

lr = linear_model.LogisticRegression()

X_train = pd.read_csv(r"D:\Internship\train_X.csv")
X_train = X_train.drop("Id", axis = 1)
X_train = np.array(X_train)

Y_train = pd.read_csv(r"D:\Internship\train_Y.csv")
Y_train = Y_train.drop("Id", axis = 1)
Y_train = np.array(Y_train)

lr.fit(X_train, Y_train)

X_test = pd.read_csv(R"D:\Internship\test_X.csv")
X_test = X_test.drop("Id", axis = 1)
age = np.mean(X_test["Age"])
fare = np.mean(X_test["Fare"])
X_test = X_test.fillna({"Age": age, "Fare": fare})
X_test = np.array(X_test)

prediction = lr.predict(X_test)

Y_test = pd.read_csv(r"D:\Internship\test_Y.csv")
Y_test = Y_test.drop("Id", axis = 1)
Y_test = np.array(Y_test)

print("Mean Squared Error: %.2f" % mean_squared_error(Y_test, prediction))
print("Coefficients of determination: %.2f" % r2_score(Y_test, prediction))
print(lr.score(X_train,Y_train))
print(lr.score(X_test, Y_test))
print(accuracy_score(Y_test, prediction))

plt.figure()
plt.plot(X_test[:,0], prediction, color = "lavender", label = "Predictions")
plt.show()
