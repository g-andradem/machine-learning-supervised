from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
from sklearn.model_selection import train_test_split
import numpy as np

def objective(x, y):
    return (x**2 + y - 11)**2 + (x + y**2 - 7)**2

def main_test():
    # DEFINE RANGE FOR INPUT
    r_min, r_max = -5.0, 5.0

    # SAMPLE INPUT RANGE UNIFORMLY AT 0.1 INCREMENTS
    xaxis = np.arange(r_min, r_max, 0.1)
    yaxis = np.arange(r_min, r_max, 0.1)

    # GP KERNEL
    kernel02 = 30*DotProduct(1.0, (1e-3, 1e3)) + 100 * WhiteKernel(10.0, (1e-3, 1e3)) + 1 * RBF(length_scale=1.0)

    gp = GPR(kernel = kernel02, alpha = 1e-8, n_restarts_optimizer = 10, random_state = 80)

    data_list = [[0 for i in range(2)] for j in range(len(xaxis))]
    z_list = [0 for i in range(len(xaxis))]

    # print('data_list:', data_list, len(data_list))
    # print('z_list:', z_list, len(z_list))

    for i in range(len(data_list)):
        data_list[i][0] = round(xaxis[i], 2)
        data_list[i][1] = round(yaxis[i], 2)
        z_list[i] = objective(round(xaxis[i], 2), round(yaxis[i], 2))

    # print('data_list_before:', data_list, np.array(data_list).shape)
    # print('len(z_list):', len(z_list))

    x_train, x_test, y_train, y_test = train_test_split(data_list, z_list, test_size = 0.80, random_state = 42)

    gp.fit(x_train, y_train)

    print("R2 of predicted 'all data' from data_list: {:.4f}".format(gp.score(data_list, z_list)))
    print("R2 of predicted 'train' from x_train: {:.4f}".format(gp.score(x_train, y_train)))
    print("R2 of predicted 'test' from y_test: {:.4f}".format(gp.score(x_test, y_test)))


if __name__ == '__main__':
    main_test()