"""
  Harness to manage optimisation domains.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=arguments-differ
# pylint: disable=abstract-class-not-used

import numpy as np
# Local
from nas.nasbot.opt.function_caller import FunctionCaller
from nas.nasbot.opt.ga_optimiser import ga_opt_args, ga_optimise_from_args
from nas.nasbot.gp.kernel import SEKernel
from nas.nasbot.utils.oper_utils import random_maximise, direct_ft_maximise
from nas.nasbot.utils.option_handler import load_options
from nas.nasbot.utils.reporters import get_reporter
from nas.nasbot.opt.worker_manager import SyntheticWorkerManager


_EUCLIDEAN_DFLT_OPT_METHOD = 'rand'
_NN_DFLT_OPT_METHOD = 'ga'


class Domain(object):
  """ Domain class. An abstract class which implements domains. """

  def __init__(self, dflt_domain_opt_method):
    """ Constructor. """
    super(Domain, self).__init__()
    self.dflt_domain_opt_method = dflt_domain_opt_method

  def maximise_obj(self, opt_method, obj, num_evals, *args, **kwargs):
    """ Optimises the objective and returns it. """
    if opt_method == 'dflt_domain_opt_method':
      opt_method = self.dflt_domain_opt_method
    return self._child_maximise_obj(opt_method, obj, num_evals, *args, **kwargs)

  def _child_maximise_obj(self, opt_method, obj, num_evals, *args, **kwargs):
    """ Child class implementation for optimising an objective. """
    raise NotImplementedError('Implement in a child class.')

  def get_default_kernel(self, *args, **kwargs):
    """ Get the default kernel for this domain. """
    raise NotImplementedError('Implement in a child class.')

  def get_type(self):
    """ Returns the type of the domain. """
    raise NotImplementedError('Implement in a child class.')

  def get_dim(self):
    """ Returns the dimension of the space. """
    raise NotImplementedError('Implement in a child class.')


# For euclidean spaces ---------------------------------------------------------------
class EuclideanDomain(Domain):
  """ Domain for Euclidean spaces. """

  def __init__(self, bounds):
    """ Constructor. """
    self.bounds = np.array(bounds)
    self._dim = len(bounds)
    super(EuclideanDomain, self).__init__('rand')

  def _child_maximise_obj(self, opt_method, obj, num_evals):
    """ Child class implementation for optimising an objective. """
    if opt_method == 'rand':
      return self._rand_maximise_obj(obj, num_evals)
    elif opt_method == 'direct':
      return self._direct_maximise_obj(obj, num_evals)
    else:
      raise ValueError('Unknown opt_method=%s for EuclideanDomain'%(opt_method))

  def _rand_maximise_obj(self, obj, num_evals):
    """ Maximise with random evaluations. """
    if num_evals is None:
      lead_const = 10 * min(5, self.dim)**2
      num_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)
    opt_val, opt_pt = random_maximise(obj, self.bounds, num_evals)
    return opt_val, opt_pt

  def _direct_maximise_obj(self, obj, num_evals):
    """ Maximise with direct. """
    if num_evals is None:
      lead_const = 10 * min(5, self.dim)**2
      num_evals = lambda t: np.clip(lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)
    lb = self.bounds[:, 0]
    ub = self.bounds[:, 1]
    opt_val, opt_pt, _ = direct_ft_maximise(obj, lb, ub, num_evals)
    return opt_val, opt_pt

  def get_default_kernel(self, range_Y):
    """ Returns the default (SE) kernel. """
    return SEKernel(self.dim, range_Y/4.0, dim_bandwidths=0.05*np.sqrt(self.dim))

  def get_type(self):
    """ Returns the type of the domain. """
    return 'euclidean'

  def get_dim(self):
    """ Return the dimensions. """
    return self._dim


class NNDomain(Domain):
  """ Domain for Neural Network Architectures. """

  def __init__(self, nn_type=None, constraint_checker=None):
    """ A domain object for neural networks. """
    self.constraint_checker = constraint_checker
    self.nn_type = nn_type
    super(NNDomain, self).__init__('ga')

  def _child_maximise_obj(self, opt_method, obj, num_evals, *args, **kwargs):
    """ Child class implementation for optimising an objective. """
    if opt_method == 'ga':
      return self._ga_maximise(obj, num_evals, *args, **kwargs)
    elif opt_method == 'rand_ga':
      return self._rand_ga_maximise(obj, num_evals)
    else:
      raise ValueError('Unknown method=%s for NNDomain'%(opt_method))

  @classmethod
  def _get_ga_optimiser_args(cls, obj, num_evals, mutation_op, init_pool,
                             init_pool_vals=None, expects_inputs_to_be_iterable=True):
    """ Returns arguments for the optimiser. """
    if expects_inputs_to_be_iterable:
      def _obj_wrap(_obj, _x):
        """ A wrapper for the optimiser for GA. """
        ret = _obj([_x])
        return ret[0]
      def _get_obj_wrap(_obj):
        """ Returns an optimiser for GA. """
        return lambda x: _obj_wrap(_obj, x)
      obj = _get_obj_wrap(obj)
    if init_pool_vals is None:
      init_pool_vals = [obj(nn) for nn in init_pool]
    reporter = get_reporter('silent')
    options = load_options(ga_opt_args, reporter=reporter)
    options.pre_eval_points = init_pool
    options.pre_eval_vals = init_pool_vals
    options.pre_eval_true_vals = init_pool_vals
    options.num_mutations_per_epoch = int(np.clip(3 * np.sqrt(num_evals), 5, 100))
#     print 'GA opt args: ', num_evals, options.num_mutations_per_epoch
    return obj, options, reporter, mutation_op

  def _ga_maximise(self, obj, num_evals, mutation_op, init_pool, init_pool_vals=None,
                   expects_inputs_to_be_iterable=True):
    """ Maximise with genetic algorithms.
       if expects_inputs_as_list is True it means the function expects the inputs to
       be iterable by default.
    """
    # Prep necessary variables
    obj, options, reporter, mutation_op = self._get_ga_optimiser_args(obj, num_evals,
      mutation_op, init_pool, init_pool_vals, expects_inputs_to_be_iterable)
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    func_caller = FunctionCaller(obj, self)
    opt_val, opt_pt, _ = ga_optimise_from_args(
                           func_caller, worker_manager, num_evals, 'asy', mutation_op,
                           options=options, reporter=reporter)
    return opt_val, opt_pt

  def _rand_ga_maximise(self, obj, num_evals):
    """ Maximise over the space of neural networks via rand_ga. """
    raise NotImplementedError('Not implemented rand_ga for NNDomain yet.')

  def get_default_kernel(self, tp_comp, mislabel_coeffs, struct_coefs, powers,
                         dist_type, range_Y):
    """ Returns the default (SE) kernel. """
    raise NotImplementedError('To do.')

  def get_type(self):
    """ Returns the type of the domain. """
    return self.nn_type

  def get_dim(self):
    """ Return the dimensions. """
    return 5

