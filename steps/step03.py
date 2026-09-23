import numpy as np
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor

# Kernel
# uma função que define como o modelo enxerga a relação entre dois pontos.
# "Quanto dois pontos estão relacionados entre si?"
from sklearn.gaussian_process.kernels import RBF

# FORMULA DO RBF (Similaridade zerada 0 --- 1 Similaridade Alta)
# l = length_scale / x, x' = pontos
# k(x,x′) = exp((x−x′)^2​ / 2l^2)

def create_gauss_process(X, y):
    plt.scatter(X, y, label="Dados")

    # CREATE KERNEL
    kernel = RBF(length_scale = 10)

    # CREATE GAUSSIAN PROCESS
    GP = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    GP.fit(X.reshape(-1, 1), y)

    print("Kernel inicial:", kernel)
    print("Kernel aprendido:", GP.kernel_)

    X_test = np.linspace(0, 10, 500)
    X_test_reshape = X_test.reshape(-1,1)

    # PREDICTION AND UNCERTAINTY
    prediction, std = GP.predict(X_test_reshape, return_std=True)

    lower = prediction - 1.96 * std
    upper = prediction + 1.96 * std

    plt.plot(X_test,prediction,label="GP")

    # Coloca incerteza entre os valores de Test
    plt.fill_between(X_test, lower, upper, alpha=0.2, label="Incerteza")

    plt.xlabel("X")
    plt.ylabel("y")

    plt.legend()
    plt.show()