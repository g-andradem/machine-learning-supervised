import matplotlib.pyplot as plt

from settings import *

def plot_1(X_test, prediction, std):
    plt.figure()
    plt.title('by X')
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

def plot_2(Y_data):
    loops_list = list(range(LOOPS))

    plt.figure()
    plt.title('by Loops')
    plt.xlabel('Loops')
    plt.ylabel('Y')
    plt.plot(
        loops_list, 
        Y_data,
        label="Y_MAX_LOOP"
    )

def show_plots():
    plt.legend()
    plt.show()