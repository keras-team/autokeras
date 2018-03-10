import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel


def edit_distance(x, y):
    ret = 0
    ret += abs(x.n_conv - y.n_conv)
    ret += abs(x.n_dense - y.n_dense)

    for i in range(min(x.n_conv, y.n_conv)):
        a = x.conv_widths[i]
        b = y.conv_widths[i]
        ret += abs(a - b) / max(a, b)

    for i in range(min(x.n_dense, y.n_dense)):
        a = x.dense_widths[i]
        b = y.dense_widths[i]
        ret += abs(a - b) / max(a, b)

    for connection in x.skip_connections:
        if connection not in y.skip_connections:
            ret += 1

    return ret


class NetKernel(Kernel):

    def is_stationary(self):
        return False

    def diag(self, X):
        return [1.0] * X.shape[0]

    def __call__(self, X, Y=None, eval_gradient=False):
        if Y is None:
            return [1.0] * X.shape[0]
        ret = np.zeros(X.shape[0], Y.shape[0])
        for x_index, x in enumerate(X):
            for y_index, y in Y:
                ret[x_index][y_index] = 1.0 / np.exp(edit_distance(x, y))
        return ret


class IncrementalGaussianProcess(GaussianProcessRegressor):
    def incremental_fit(self, train_x, train_y):
        pass

    def first_fit(self, train_x, train_y):
        pass
