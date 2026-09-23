import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def create_linear_regression_2d(X, y):
    # CREATE MODEL
    model = LinearRegression()
    model.fit(X.reshape(-1, 1), y)

    X_test = np.linspace(0, 10, 200).reshape(-1, 1)
    y_pred = model.predict(X_test)

    plt.scatter(X, y, label = 'real_values')
    plt.plot(X_test, y_pred, label = 'predict + noise')

    plt.xlabel("X")
    plt.ylabel("y")

    plt.legend()
    plt.show()