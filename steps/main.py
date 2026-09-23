import numpy as np

# REDE NEURAL
# "Vou encontrar uma função que explique meus dados."

# GP
# "Vou considerar várias funções possíveis e, depois de observar os dados, determinar quais são mais plausíveis."
# "Qual é minha previsão e quão confiante estou nela?"

# SVR - Support Vector Regression
# "Quero encontrar uma função que fique dentro de uma margem de tolerância dos dados."

from step01 import create_linear_regression_1d
from step02 import create_linear_regression_2d
from step03 import create_gauss_process
from step04 import create_SVR

def linear_regression_1d():
    X = np.array([1, 2, 3, 4, 5, 6])
    y = np.array([2.1, 3.8, 5.2, 6.9, 8.1, 9.0])

    create_linear_regression_1d(X, y)

def linear_regression_2d():
    X = np.concatenate([
        np.linspace(0, 3, 20),
        np.linspace(7, 10, 20)
    ])

    # NOISE -> RUIDO (PEQUENA VARIACAO)
    # MEDIA (SOBE OU DESCE), DESVIO PADRAO (INTENSIDADE), TAMANHO
    noise = np.random.normal(0, 0.2, len(X))

    # Y = SEN DOS VALORES DE X + RUIDO
    y = np.sin(X)#  + noise

    create_linear_regression_2d(X, y)

def gauss_process():
    X = np.concatenate([
        np.linspace(0, 3, 10),
        np.linspace(7, 10, 10)
    ])

    noise = np.random.normal(0, 0.2, len(X))
    y = np.sin(X) + noise

    create_gauss_process(X, y)

def SVR():
    X = np.linspace(0, 10, 30)

    noise = np.random.normal(0, 0.2, len(X))
    y = np.sin(X) + noise

    X = X.reshape(-1,1)

    create_SVR(X, y)

def main():
    # Linear Regression so consegue fazer retas
    # linear_regression_1d()
    # linear_regression_2d()

    gauss_process()

    # SVR()

if __name__ == '__main__':
    main()