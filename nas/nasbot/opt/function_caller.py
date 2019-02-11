"""
  Harness for calling a function.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=no-member
# pylint: disable=relative-import
# pylint: disable=invalid-name

from argparse import Namespace
import numpy as np
# Local
from ..utils.general_utils import map_to_cube, map_to_bounds
# from domains import EuclideanDomain

EVAL_ERROR_CODE = 'eval_error_2401181243'


class FunctionCaller(object):
  """ The basic function caller class.
      All other function callers should inherit this class.
  """

  def __init__(self, func, domain, opt_pt=None, opt_val=None, noise_type='none',
               noise_params=None, descr=''):
    """ Constructor.
      - func: takes argument x and returns function value.
      - noise_type: A string indicating what type of noise to add. If 'none' will not
                    add any noise.
      - noise_params: Any parameters for the noise random variable.
      - raw_opt_pt, raw_opt_val: optimum point and value if known. "raw" because, when
          actually calling the function, you might want to do some normalisation of x.
    """
    self.func = func
    self.domain = domain
    self.noise_type = noise_type
    self.noise_params = noise_params
    self.descr = descr
    self.opt_pt = opt_pt     # possibly over-written by child class
    self.opt_val = opt_val   # possibly over-written by child class
    self.noise_adder = None
    self._set_up_noise_adder()

  def _set_up_noise_adder(self):
    """ Sets up a function to add noise. """
    if self.noise_type == 'none':
      self.noise_adder = lambda num_samples: np.zeros(shape=(num_samples))
    elif self.noise_type == 'gauss':
      self.noise_adder = lambda num_samples: np.random.normal(size=(num_samples))
    else:
      raise NotImplementedError(('Not implemented %s yet. Only implemented Gaussian noise'
                                 + ' so far.')%(self.noise_type))

  def eval_single(self, x, qinfo=None, noisy=True):
    """ Evaluates func at a single point x. If noisy is True and noise_type is \'none\'
        will add noise.
    """
    qinfo = Namespace() if qinfo is None else qinfo
    true_val = float(self.func(x))
    if true_val == EVAL_ERROR_CODE:
      val = EVAL_ERROR_CODE
    else:
      val = true_val if not noisy else true_val + self.noise_adder(1)[0]
    # put everything into qinfo
    qinfo.point = x
    qinfo.true_val = true_val
    qinfo.val = val
    return val, qinfo

  def eval_multiple(self, X, qinfos=None, noisy=True):
    """ Evaluates the function at a list of points in a list X.
        Creating this because when the domain is Euclidean there may be efficient
        vectorised implementations.
    """
    qinfos = [None] * len(X) if qinfos is None else qinfos
    ret_vals = []
    ret_qinfos = []
    for i in range(len(X)):
      val, qinfo = self.eval_single(X[i], qinfos[i], noisy)
      ret_vals.append(val)
      ret_qinfos.append(qinfo)
    return ret_vals, ret_qinfos


# Function Caller for Euclidean spaces ================================================
class EuclideanFunctionCaller(FunctionCaller):
  """ Function caller for Euclidean spaces. """

  def __init__(self, func, domain, vectorised, raw_opt_pt, opt_val, *args, **kwargs):
    """ Constructor. """
    if hasattr(domain, '__iter__'):
      # Then the constructor has received the bounds. Obtain a domian object from this.
      from domains import EuclideanDomain
      domain = EuclideanDomain(domain)
    self.true_dom_bounds = domain.bounds
    self.dom_bounds = np.array([[0, 1]] * domain.get_dim())
    self.vectorised = vectorised
    self.dim = domain.get_dim()
    self.raw_opt_pt = raw_opt_pt
    opt_pt = None if raw_opt_pt is None else self.get_normalised_coords(raw_opt_pt)
    super(EuclideanFunctionCaller, self).__init__(func, domain, opt_pt, opt_val,
                                                  *args, **kwargs)

  # Override tools for evaluating the function -------------------------------------------
  def eval_single(self, x, qinfo=None, normalised=True, noisy=True):
    """ Evaluates func at a single point x. If noisy is True and noise_type is not None
        then will add noise. If normalised is true, evaluates on normalised coords."""
    # pylint: disable=arguments-differ
    x_sent = x
    if normalised:
      x = self.get_unnormalised_coords(x)
    if not self.vectorised:
      func_val = float(self.func(x))
    else:
      X = np.array(x).reshape(1, self.domain.get_dim())
      func_val = float(self.func(X))
    if noisy and self.noise_type != 'none':
      ret = func_val + self.noise_adder(1)[0]
    else:
      ret = func_val
    if qinfo is None:
      qinfo = Namespace()
    qinfo.point = x_sent # Include the query point in qinfo.
    qinfo.true_val = func_val
    return ret, qinfo

  def eval_multiple(self, X, qinfo=None, normalised=True, noisy=True):
    """ Evaluates func at multiple points. """
    # pylint: disable=arguments-differ
    # Check the dat namespace
    if qinfo is None:
      qinfo = Namespace()
    if normalised:
      X = self.get_unnormalised_coords(X)
    if self.vectorised:
      func_vals = self.func(X).ravel()
    else:
      ret = []
      for i in range(len(X)):
        ret_val, _ = self.eval_single(X[i, :])
        ret.append(ret_val)
      func_vals = np.array(ret)
    if noisy and self.noise_type != 'none':
      ret = func_vals + self.noise_adder(len(X))
    else:
      ret = func_vals
    qinfo.points = X # Include the query point in qinfo.
    qinfo.true_vals = func_vals
    return ret, qinfo

  # Map to normalised coordinates and vice versa -----------------------------------------
  def get_normalised_coords(self, X):
    """ Maps points in the original space to the unit cube. """
    return map_to_cube(X, self.true_dom_bounds)

  def get_unnormalised_coords(self, X):
    """ Maps points in the unit cube to the orignal space. """
    return map_to_bounds(X, self.true_dom_bounds)


# Some APIs ------------------------------------------------
def get_euc_function_caller_from_function(func, domain_bounds, vectorised, opt_pt=None,
                                          opt_val=None, *args, **kwargs):
  """ Just calls EuclideanFunctionCaller. Keeping around for legacy issues. """
  return EuclideanFunctionCaller(func, domain_bounds, vectorised,
                                 raw_opt_pt=opt_pt, opt_val=opt_val, *args, **kwargs)

