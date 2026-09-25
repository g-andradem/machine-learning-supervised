from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

import numpy as np

from settings import *
from plots import *

def read_data(name_file):
    with open(f"mini-project/{name_file}", 'r') as file:
        pass

def loop_fit_prediction(gp, X, y, X_test):
    gp.fit(X, y)

    prediction, std = gp.predict(X_test, return_std = True)

    return prediction, std

# BLACK BOX
def black_box(X):
    # noise = np.random.normal(0, 0.05, np.shape(x))
    x1 = X[:, 0]
    x2 = X[:, 1]

    return np.sin(x1) + 0.5 * np.sin(3*x2) + 0.1 * x1 + 0.2 * x2 # + noise

def main():

    # Data Initial
    X = np.array([[0, 0], [3, 2], [8, 5], [13, 7], [30, 10]])
    y = black_box(X)

    # Gaussian Process
    kernel = RBF(LENGTH_SCALE)
    gp = GaussianProcessRegressor(kernel = kernel,optimizer = None)

    # Data Test
    x1 = np.linspace(MIN, MAX, POINTS_NUMBER)
    x2 = np.linspace(MIN, MAX, POINTS_NUMBER)
    X1, X2 = np.meshgrid(x1, x2)
    X_test = np.column_stack([X1.ravel(), X2.ravel()])

    # Write Data
    Y_data = []

    for loop in range(LOOPS):
        prediction, std = loop_fit_prediction(gp, X, y, X_test)

        BETA = 5 * (1 - loop / LOOPS)
        ucb = prediction + BETA * std
        best_index = np.argmax(ucb)
        next_x = X_test[best_index]
        next_y_pred = prediction[best_index]
        real_y = black_box(next_x.reshape(1, -1))[0]

        X = np.vstack([X, next_x])
        y = np.append(y, real_y)

        Y_data.append(np.max(y))

    # plot_1(X_test, prediction, std) # Plot 1 by X    
    plot_2(Y_data) # Plot 2 by Loops

    show_plots()

if __name__ == '__main__':
    main()