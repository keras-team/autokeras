import numpy as np
from scipy.linalg import cholesky, cho_solve


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


class IncrementalGaussianProcess:
    def __init__(self):
        self.alpha = 1e-10
        self._k_matrix = None
        self._x = None
        self._y = None
        self._first_fitted = False
        self._l_matrix = None
        self._alpha_vector = None
        self.kernel = kernel

    def incremental_fit(self, train_x, train_y):
        if not self._first_fitted:
            raise ValueError("The first_fit function needs to be called first.")

        train_x, train_y = np.array([train_x]), np.array([train_y])

        # Incrementally compute K
        up_right_k = self.kernel(self._x, train_x)  # Shape (len(X_train_), len(train_x))
        down_left_k = np.transpose(up_right_k)
        down_right_k = self.kernel(train_x)
        down_right_k[np.diag_indices_from(down_right_k)] += self.alpha
        up_k = np.concatenate((self._k_matrix, up_right_k), axis=1)
        down_k = np.concatenate((down_left_k, down_right_k), axis=1)
        self._k_matrix = np.concatenate((up_k, down_k), axis=0)

        self._x = np.concatenate((self._x, train_x), axis=0)
        self._y = np.concatenate((self._y, train_y), axis=0)

        self._l_matrix = cholesky(self._k_matrix, lower=True)  # Line 2

        self._alpha_vector = cho_solve((self._l_matrix, True), self._y)  # Line 3

        return self

    def first_fit(self, train_x, train_y):
        train_x, train_y = np.array([train_x]), np.array([train_y])

        self._x = np.copy(train_x)
        self._y = np.copy(train_y)

        self._k_matrix = self.kernel(self._x)
        self._k_matrix[np.diag_indices_from(self._k_matrix)] += self.alpha

        self._l_matrix = cholesky(self._k_matrix, lower=True)  # Line 2

        self._alpha_vector = cho_solve((self._l_matrix, True), self._y)  # Line 3

        self._first_fitted = True
        return self

    def predict(self, train_x):
        k_trans = self.kernel(train_x, self._x)
        y_mean = k_trans.dot(self._alpha_vector)  # Line 4 (y_mean = f_star)
        return y_mean


def kernel(X, Y=None):
    if Y is None:
        ret = np.zeros((X.shape[0], X.shape[0]))
        for x_index, x in enumerate(X):
            for y_index, y in enumerate(X):
                if x_index == y_index:
                    ret[x_index][y_index] = 1.0
                elif x_index < y_index:
                    ret[x_index][y_index] = 1.0 / np.exp(edit_distance(x, y))
                else:
                    ret[x_index][y_index] = ret[y_index][x_index]
        return ret
    ret = np.zeros((X.shape[0], Y.shape[0]))
    for x_index, x in enumerate(X):
        for y_index, y in enumerate(Y):
            ret[x_index][y_index] = 1.0 / np.exp(edit_distance(x, y))
    return ret
