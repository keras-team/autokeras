import warnings

import numpy as np
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import linear_sum_assignment


def layer_distance(a, b):
    return abs(a - b) * 1.0 / max(a, b)


def layers_distance(list_a, list_b):
    len_a = len(list_a)
    len_b = len(list_b)
    f = np.zeros((len_a + 1, len_b + 1))
    f[-1][-1] = 0
    for i in range(-1, len_a):
        f[i][-1] = i + 1
    for j in range(-1, len_b):
        f[-1][j] = j + 1
    for i in range(len_a):
        for j in range(len_b):
            f[i][j] = min(f[i][j - 1] + 1, f[i - 1][j] + 1, f[i - 1][j - 1] + layer_distance(list_a[i], list_b[j]))
    return f[len_a][len_b]


def skip_connection_distance(a, b):
    if a[2] != b[2]:
        return 1.0
    len_a = abs(a[1] - a[0])
    len_b = abs(b[1] - b[0])
    return (abs(a[0] - b[0]) + abs(len_a - len_b)) / (max(a[0], b[0]) + max(len_a, len_b))


def skip_connections_distance(list_a, list_b):
    distance_matrix = np.zeros((len(list_a), len(list_b)))
    for i, a in enumerate(list_a):
        for j, b in enumerate(list_b):
            distance_matrix[i][j] = skip_connection_distance(a, b)
    return distance_matrix[linear_sum_assignment(distance_matrix)].sum() + abs(len(list_a) - len(list_b))


def edit_distance(x, y):
    ret = 0
    ret += layers_distance(x.conv_widths, y.conv_widths)
    ret += layers_distance(x.dense_widths, y.dense_widths)
    ret += skip_connections_distance(x.skip_connections, y.skip_connections)
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

    @property
    def first_fitted(self):
        return self._first_fitted

    def first_fit(self, train_x, train_y):
        train_x, train_y = np.array(train_x), np.array(train_y)

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

        # compute inverse K_inv of K based on its Cholesky
        # decomposition L and its inverse L_inv
        l_inv = solve_triangular(self._l_matrix.T, np.eye(self._l_matrix.shape[0]))
        k_inv = l_inv.dot(l_inv.T)
        # Compute variance of predictive distribution
        y_var = np.ones(len(train_x), dtype=np.float)
        y_var -= np.einsum("ij,ij->i", np.dot(k_trans, k_inv), k_trans)

        # Check if any of the variances is negative because of
        # numerical issues. If yes: set the variance to 0.
        y_var_negative = y_var < 0
        if np.any(y_var_negative):
            warnings.warn("Predicted variances smaller than 0. "
                          "Setting those variances to 0.")
            y_var[y_var_negative] = 0.0
        return y_mean, np.sqrt(y_var)


def kernel(train_x, train_y=None):
    if train_y is None:
        ret = np.zeros((train_x.shape[0], train_x.shape[0]))
        for x_index, x in enumerate(train_x):
            for y_index, y in enumerate(train_x):
                if x_index == y_index:
                    ret[x_index][y_index] = 1.0
                elif x_index < y_index:
                    ret[x_index][y_index] = 1.0 / np.exp(edit_distance(x, y))
                else:
                    ret[x_index][y_index] = ret[y_index][x_index]
        return ret
    ret = np.zeros((train_x.shape[0], train_y.shape[0]))
    for x_index, x in enumerate(train_x):
        for y_index, y in enumerate(train_y):
            ret[x_index][y_index] = 1.0 / np.exp(edit_distance(x, y))
    return ret
