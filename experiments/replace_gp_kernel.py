import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel


class DotProduct(Kernel):
    """Dot-Product kernel.

    The DotProduct kernel is non-stationary and can be obtained from linear
    regression by putting N(0, 1) priors on the coefficients of x_d (d = 1, . .
    . , D) and a prior of N(0, \sigma_0^2) on the bias. The DotProduct kernel
    is invariant to a rotation of the coordinates about the origin, but not
    translations. It is parameterized by a parameter sigma_0^2. For
    sigma_0^2 =0, the kernel is called the homogeneous linear kernel, otherwise
    it is inhomogeneous. The kernel is given by

    k(x_i, x_j) = sigma_0 ^ 2 + x_i \cdot x_j

    The DotProduct kernel is commonly combined with exponentiation.

    .. versionadded:: 0.18

    Parameters
    ----------
    sigma_0 : float >= 0, default: 1.0
        Parameter controlling the inhomogenity of the kernel. If sigma_0=0,
        the kernel is homogenous.

    sigma_0_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on l

    """

    def __init__(self, sigma_0=1.0, sigma_0_bounds=(1e-5, 1e5)):
        self.sigma_0 = sigma_0
        self.sigma_0_bounds = sigma_0_bounds

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.

        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.

        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)

        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """
        if eval_gradient:
            raise ValueError(
                "Gradient can only be evaluated when Y is None.")
        X = np.atleast_2d(X)
        if Y is None:
            K = np.inner(X, X) + self.sigma_0 ** 2
        else:
            K = np.inner(X, Y) + self.sigma_0 ** 2

        return K

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.einsum('ij,ij->i', X, X) + self.sigma_0 ** 2

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return False


gp = GaussianProcessRegressor(kernel=DotProduct())
x = np.random.rand(100, 5)
y = np.random.rand(100, 1)
gp.fit(x, y)
gp.predict(x)
