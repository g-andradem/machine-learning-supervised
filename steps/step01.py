import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def create_linear_regression_1d(X, y):
    # CREATE LINEAR REGRESSION MODEL
    model = LinearRegression()

    # TRIES TO LEARN THE MODEL
    model.fit(X.reshape(-1, 1), y)

    # CREATE PREDICTION
    # O np.linspace() cria números igualmente espaçados.
    # .reshape(-1, 1) transforma no formato que o scikit-learn aceita
    X_test = np.linspace(1, 6, 100).reshape(-1, 1)
    y_pred = model.predict(X_test)

    # PLOT VALUES
    # REAL VALUES (POINTS)
    plt.scatter(X, y, label = 'real_values')

    # PREDICT VALUES (LINE)
    plt.plot(X_test, y_pred, label = 'predict')

    # SET NAMES LABELS
    plt.xlabel('horas estudadas')
    plt.ylabel('nota')

    # SHOW
    plt.legend()
    plt.show()