import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR

def create_SVR(X, y):

    # kernel = "linear"
    # ε pequeno - tenta acompanhar mais os dados - modelo pode ficar mais curvado/complexo.
    # ε grande - ignora diferenças pequenas - modelo tende a ficar mais simples/suave.
    # C pequeno - o modelo tolera mais erros - curva mais simples.
    # C grande - o modelo tenta evitar erros - pode ficar mais ajustado aos dados.
    # gamma pequeno → influência mais ampla → curva mais suave
    # gamma grande → influência mais local → curva pode ficar bem mais ondulada
    models = [
        SVR(kernel="rbf", gamma = 0.1),
        SVR(kernel="rbf", gamma = 1),
        SVR(kernel="rbf", gamma = 10)
    ]

    for model in models:
        model.fit(X, y)

    X_test = np.linspace(0, 10, 500).reshape(-1, 1)

    predictions = [model.predict(X_test) for model in models]

    plt.scatter(X, y, label = 'data')
    # epsilon / ε = 0.01, ε = 0.2, ε = 1.0
    # C / C = 0.1, C = 1.0, C = 100
    plt.plot(X_test, predictions[0], label = "gamma = 0.1")
    plt.plot(X_test, predictions[1], label = "gamma = 1")
    plt.plot(X_test, predictions[2], label = "gamma = 10")

    plt.xlabel('X')
    plt.ylabel('y')

    plt.legend()
    plt.show()