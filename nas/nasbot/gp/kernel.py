"""
  Implements various kernels.
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np

# Local imports
from nas.nasbot.utils.general_utils import dist_squared


class Kernel(object):
  """ A kernel class. """

  def __init__(self):
    """ Constructor. """
    super(Kernel, self).__init__()
    self.hyperparams = {}

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    raise NotImplementedError('Implement in a child class.')

  def __call__(self, X1, X2=None):
    """ Evaluates the kernel by calling evaluate. """
    return self.evaluate(X1, X2)

  def evaluate(self, X1, X2=None):
    """ Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is a wrapper for _child_evaluate.
    """
    X2 = X1 if X2 is None else X2
    return self._child_evaluate(X1, X2)

  def evaluate_from_dists(self, dists):
    """ Evaluates the kernel from pairwise distances. """
    raise NotImplementedError('Implement in a child class.')

  def _child_evaluate(self, X1, X2):
    """ Evaluates kernel values between X1 and X2 and returns an n1xn2 kernel matrix.
        This is to be implemented in a child kernel.
    """
    raise NotImplementedError('Implement in a child class.')

  def set_hyperparams(self, **kwargs):
    """ Set hyperparameters here. """
    self.hyperparams = kwargs

  def add_hyperparams(self, **kwargs):
    """ Set additional hyperparameters here. """
    for key, value in kwargs.iteritems():
      self.hyperparams[key] = value


class SEKernel(Kernel):
  """ Squared exponential kernel. """

  def __init__(self, dim, scale=None, dim_bandwidths=None):
    """ Constructor. dim is the dimension. """
    super(SEKernel, self).__init__()
    self.dim = dim
    self.set_se_hyperparams(scale, dim_bandwidths)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

  def set_dim_bandwidths(self, dim_bandwidths):
    """ Sets the bandwidth for each dimension. """
    if dim_bandwidths is not None:
      if len(dim_bandwidths) != self.dim:
        raise ValueError('Dimension of dim_bandwidths should be the same as dimension.')
      dim_bandwidths = np.array(dim_bandwidths)
    self.add_hyperparams(dim_bandwidths=dim_bandwidths)

  def set_single_bandwidth(self, bandwidth):
    """ Sets the bandwidht of all dimensions to be the same value. """
    dim_bandwidths = None if bandwidth is None else [bandwidth] * self.dim
    self.set_dim_bandwidths(dim_bandwidths)

  def set_scale(self, scale):
    """ Sets the scale parameter for the kernel. """
    self.add_hyperparams(scale=scale)

  def set_se_hyperparams(self, scale, dim_bandwidths):
    """ Sets both the scale and the dimension bandwidths for the SE kernel. """
    self.set_scale(scale)
    if hasattr(dim_bandwidths, '__len__'):
      self.set_dim_bandwidths(dim_bandwidths)
    else:
      self.set_single_bandwidth(dim_bandwidths)

  def evaluate_from_dists(self, dists):
    """ Evaluates the kernel from pairwise distances. """
    raise NotImplementedError('Not implemented yet.')

  def _child_evaluate(self, X1, X2):
    """ Evaluates the SE kernel between X1 and X2 and returns the gram matrix. """
    scaled_X1 = self.get_scaled_repr(X1)
    scaled_X2 = self.get_scaled_repr(X2)
    dist_sq = dist_squared(scaled_X1, scaled_X2)
    K = self.hyperparams['scale'] * np.exp(-dist_sq/2)
    return K

  def get_scaled_repr(self, X):
    """ Returns the scaled version of an input by the bandwidths. """
    return X/self.hyperparams['dim_bandwidths']

  def get_effective_norm(self, X, order=None, is_single=True):
    """ Gets the effective norm. That is the norm of X scaled by bandwidths. """
    # pylint: disable=arguments-differ
    scaled_X = self.get_scaled_repr(X)
    if is_single:
      return np.linalg.norm(scaled_X, ord=order)
    else:
      return np.array([np.linalg.norm(sx, ord=order) for sx in scaled_X])

  def compute_std_slack(self, X1, X2):
    """ Computes a bound on the maximum standard deviation diff between X1 and X2. """
    k_12 = np.array([float(self.evaluate(X1[i].reshape(1, -1), X2[i].reshape(1, -1)))
                     for i in range(len(X1))])
    return np.sqrt(self.hyperparams['scale'] - k_12)

  def change_smoothness(self, factor):
    """ Decreases smoothness by the given factor. """
    self.hyperparams['dim_bandwidths'] *= factor


class PolyKernel(Kernel):
  """ The polynomial kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, order, scale, dim_scalings=None):
    """ Constructor. """
    super(PolyKernel, self).__init__()
    self.dim = dim
    self.set_poly_hyperparams(order, scale, dim_scalings)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

  def set_order(self, order):
    """ Sets the order of the polynomial. """
    self.add_hyperparams(order=order)

  def set_scale(self, scale):
    """ Sets the scale of the kernel. """
    self.add_hyperparams(scale=scale)

  def set_dim_scalings(self, dim_scalings):
    """ Sets the scaling for each dimension in the polynomial kernel. This will be a
        dim+1 dimensional vector.
    """
    if dim_scalings is not None:
      if len(dim_scalings) != self.dim:
        raise ValueError('Dimension of dim_scalings should be dim + 1.')
      dim_scalings = np.array(dim_scalings)
    self.add_hyperparams(dim_scalings=dim_scalings)

  def set_single_scaling(self, scaling):
    """ Sets the same scaling for all dimensions. """
    if scaling is None:
      self.set_dim_scalings(None)
    else:
      self.set_dim_scalings([scaling] * self.dim)

  def set_poly_hyperparams(self, order, scale, dim_scalings):
    """Sets the hyper parameters. """
    self.set_order(order)
    self.set_scale(scale)
    if hasattr(dim_scalings, '__len__'):
      self.set_dim_scalings(dim_scalings)
    else:
      self.set_single_scaling(dim_scalings)

  def evaluate_from_dists(self, dists):
    """ Evaluates the kernel from pairwise distances. """
    raise NotImplementedError('evaluate_from_dists not applicable for PolyKernel.')

  def _child_evaluate(self, X1, X2):
    """ Evaluates the polynomial kernel and returns and the gram matrix. """
    X1 = X1 * self.hyperparams['dim_scalings']
    X2 = X2 * self.hyperparams['dim_scalings']
    K = self.hyperparams['scale'] * ((X1.dot(X2.T) + 1)**self.hyperparams['order'])
    return K


class UnscaledPolyKernel(Kernel):
  """ The polynomial kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, order, dim_scalings=None):
    """ Constructor. """
    super(UnscaledPolyKernel, self).__init__()
    self.dim = dim
    self.set_unscaled_poly_hyperparams(order, dim_scalings)

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return True

  def set_order(self, order):
    """ Sets the order of the polynomial. """
    self.add_hyperparams(order=order)

  def set_dim_scalings(self, dim_scalings):
    """ Sets the scaling for each dimension in the polynomial kernel. This will be a
        dim+1 dimensional vector.
    """
    if dim_scalings is not None:
      if len(dim_scalings) != self.dim + 1:
        raise ValueError('Dimension of dim_scalings should be dim + 1.')
      dim_scalings = np.array(dim_scalings)
    self.add_hyperparams(dim_scalings=dim_scalings)

  def set_single_scaling(self, scaling):
    """ Sets the same scaling for all dimensions. """
    if scaling is None:
      self.set_dim_scalings(None)
    else:
      self.set_dim_scalings([scaling] * (self.dim + 1))

  def set_unscaled_poly_hyperparams(self, order, dim_scalings):
    """Sets the hyper parameters. """
    self.set_order(order)
    if hasattr(dim_scalings, '__len__'):
      self.set_dim_scalings(dim_scalings)
    else:
      self.set_single_scaling(dim_scalings)

  def evaluate_from_dists(self, dists):
    """ Evaluates the kernel from pairwise distances. """
    raise NotImplementedError('evaluate_from_dists not applicable for ' +
                              'UnscaledPolyKernel.')

  def _child_evaluate(self, X1, X2):
    """ Evaluates the polynomial kernel and returns and the gram matrix. """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    X1 = np.concatenate((np.ones((n1, 1)), X1), axis=1) * self.hyperparams['dim_scalings']
    X2 = np.concatenate((np.ones((n2, 1)), X2), axis=1) * self.hyperparams['dim_scalings']
    K = (X1.dot(X2.T))**self.hyperparams['order']
    return K


class CoordinateProductKernel(Kernel):
  """ Implements a coordinatewise product kernel. """
  # pylint: disable=abstract-method

  def __init__(self, dim, scale, kernel_list=None, coordinate_list=None):
    """ Constructor.
        kernel_list is a list of n Kernel objects. coordinate_list is a list of n lists
        each indicating the coordinates each kernel in kernel_list should be applied to.
    """
    super(CoordinateProductKernel, self).__init__()
    self.dim = dim
    self.scale = scale
    self.kernel_list = kernel_list
    self.coordinate_list = coordinate_list

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return all([kern.is_guaranteed_psd() for kern in self.kernel_list])

  def set_kernel_list(self, kernel_list):
    """ Sets a new list of kernels. """
    self.kernel_list = kernel_list

  def set_new_kernel(self, kernel_idx, new_kernel):
    """ Sets new_kernel to kernel_list[kernel_idx]. """
    self.kernel_list[kernel_idx] = new_kernel

  def set_kernel_hyperparams(self, kernel_idx, **kwargs):
    """ Sets the hyper-parameters for kernel_list[kernel_idx]. """
    self.kernel_list[kernel_idx].set_hyperparams(**kwargs)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the combined kernel. """
    n1 = X1.shape[0]
    n2 = X2.shape[0]
    K = self.scale * np.ones((n1, n2))
    for idx, kernel in enumerate(self.kernel_list):
      X1_sel = X1[:, self.coordinate_list[idx]]
      X2_sel = X2[:, self.coordinate_list[idx]]
      K *= kernel(X1_sel, X2_sel)
    return K


class ExpSumOfDistsKernel(Kernel):
  """ Given a function that returns a list of distances d1, d2, ... dk, this kernel
      takes the form exp(beta1*d1^p + beta2*d2^p + ... + betak*dk^p. """
  # pylint: disable=abstract-method

  def __init__(self, dist_computer, betas, scale, powers=1, num_dists=None,
               dist_is_hilbertian=False):
    """ Constructor.
          trans_dist_computer: Given two lists of networks X1 and X2, trans_dist_computer
            is a function which returns a list of n1xn2 matrices where ni=len(Xi).
    """
    super(ExpSumOfDistsKernel, self).__init__()
    self.num_dists = num_dists if num_dists is not None else len(betas)
    self.dist_computer = dist_computer
    betas = betas if hasattr(betas, '__iter__') else [betas] * self.num_dists
    powers = powers if hasattr(powers, '__iter__') else [powers] * self.num_dists
    self.add_hyperparams(betas=np.array(betas), powers=np.array(powers), scale=scale)
    self.dist_is_hilbertian = dist_is_hilbertian

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return self.dist_is_hilbertian

  def evaluate_from_dists(self, list_of_dists):
    """ Evaluates the kernel from pairwise distances. """
    sum_of_dists = _compute_raised_scaled_sum(list_of_dists, self.hyperparams['betas'],
                                                             self.hyperparams['powers'])
    return self.hyperparams['scale'] * np.exp(-sum_of_dists)

  def _child_evaluate(self, X1, X2):
    """ Evaluates the kernel between X1 and X2. """
    list_of_dists = self.dist_computer(X1, X2)
    return self.evaluate_from_dists(list_of_dists)

class SumOfExpSumOfDistsKernel(Kernel):
  """ Given a function that returns a list of distances d1, d2, ... dk, this kernel
      takes the form alpha_1*exp(beta11*d1^p + ... + beta1k*d2^p) +
      alpha_2 * exp(beta_21 + ... + beta2k*dk^p) + ... .
  """

  def __init__(self, dist_computer, alphas, groups, betas, powers, num_dists=None,
               dist_is_hilbertian=False):
    super(SumOfExpSumOfDistsKernel, self).__init__()
    self.num_dists = num_dists if num_dists is not None else len(betas)
    self.dist_computer = dist_computer
    assert len(alphas) == len(groups)
    betas = betas if hasattr(betas, '__iter__') else [betas] * self.num_dists
    powers = powers if hasattr(powers, '__iter__') else [powers] * self.num_dists
    self.add_hyperparams(betas=np.array(betas), powers=np.array(powers), alphas=alphas,
                         groups=groups)
    self.dist_is_hilbertian = dist_is_hilbertian

  def is_guaranteed_psd(self):
    """ The child class should implement this method to indicate whether it is
        guaranteed to be PSD. """
    return self.dist_is_hilbertian

  def evaluate_from_dists(self, list_of_dists):
    """ Evaluates the kernel from pairwise distances. """
    individual_kernels = []
    for gi, group in enumerate(self.hyperparams['groups']):
      curr_list_of_dists = [list_of_dists[i] for i in group]
      curr_betas = [self.hyperparams['betas'][i] for i in group]
      curr_powers = [self.hyperparams['powers'][i] for i in group]
      curr_sum_of_dists = _compute_raised_scaled_sum(curr_list_of_dists,
                                                     curr_betas, curr_powers)
      individual_kernels.append(self.hyperparams['alphas'][gi] *
                                np.exp(-curr_sum_of_dists))
    return sum(individual_kernels)

  def _child_evaluate(self, X1, X2):
    list_of_dists = self.dist_computer(X1, X2)
    return self.evaluate_from_dists(list_of_dists)

# Ancillary functions for the ExpSumOfDists and SumOfExpSumOfDistsKernel classes =========
def _compute_raised_scaled_sum(dist_arrays, betas, powers):
  """ Returns the distances raised to the powers and scaled by betas. """
  sum_of_dists = np.zeros(dist_arrays[0].shape)
  for idx, curr_dist in enumerate(dist_arrays):
    if powers[idx] == 1:
      raised_curr_dists = curr_dist
    else:
      raised_curr_dists = curr_dist ** powers[idx]
    sum_of_dists += betas[idx] * raised_curr_dists
  return sum_of_dists

