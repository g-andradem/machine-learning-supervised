import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from settings import *

def write_data(name_file, loop, next_x, next_y_pred, real_y):
    with open(f"mini-project/{name_file}", 'a') as file:
        file.write(f"Loop: {loop + 1}\n")
        file.write(f"Próximo x: {next_x}\n")
        file.write(f"y previsto: {next_y_pred}\n")
        file.write(f"y real: {real_y}\n")
        file.write("-" * 30 + "\n")

def black_box(x):
    return np.sin(x) + 0.3 * np.sin(3*x) + 0.1 * x

def loop_fit_prediction(gp, X, y, X_test):
    gp.fit(X.reshape(-1,1), y)

    prediction, std = gp.predict(X_test, return_std = True)

    return prediction, std

def main():

    # Data Initial
    X = np.array([0, 2, 4, 7, 10])
    y = black_box(X)

    # Gaussian Process
    kernel = RBF(length_scale = 1.0)
    gp = GaussianProcessRegressor(kernel = kernel,optimizer = None)

    # Data Test
    X_test = np.linspace(0, 10, 500).reshape(-1, 1)

    for loop in range(5):
        prediction, std = loop_fit_prediction(gp, X, y, X_test)

        ucb = prediction + BETA * std
        best_index = np.argmax(ucb)
        next_x = X_test[best_index][0]

        next_x = X_test[best_index][0]
        next_y_pred = prediction[best_index]
        real_y = black_box(next_x)

        write_data(
            'data.txt',
            loop,
            next_x, 
            next_y_pred, 
            real_y
        )

        X = np.append(X, next_x)
        y = np.append(y, real_y)

    plt.xlabel('X')
    plt.ylabel('Y')

    plt.plot(
        X_test, 
        prediction, 
        label="GP"
    )

    plt.fill_between(
        X_test.ravel(), 
        prediction - 1.96 * std, 
        prediction + 1.96 * std, 
        alpha = 0.2, 
        label="Incerteza"
    )

    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()