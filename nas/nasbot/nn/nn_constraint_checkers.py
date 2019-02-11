"""
  Harness to check if a neural network satisfies certain constriants. Mostly needed to
  constrain the search space when optimising them.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments

import numpy as np
# Local
from ..opt.domains import NNDomain


class NNConstraintChecker(object):
  """ A class for checking if a neural network satisfies constraints. """

  def __init__(self, max_num_layers, min_num_layers, max_mass, min_mass,
               max_in_degree, max_out_degree, max_num_edges,
               max_num_units_per_layer, min_num_units_per_layer):
    """ Constructor. """
    super(NNConstraintChecker, self).__init__()
    self.max_num_layers = max_num_layers
    self.min_num_layers = min_num_layers
    self.max_mass = max_mass
    self.min_mass = min_mass
    self.max_in_degree = max_in_degree
    self.max_out_degree = max_out_degree
    self.max_num_edges = max_num_edges
    self.max_num_units_per_layer = max_num_units_per_layer
    self.min_num_units_per_layer = min_num_units_per_layer

  def __call__(self, nn, *args, **kwargs):
    """ Checks if the constraints are satisfied for the given nn. """
    return self.constraints_are_satisfied(nn, *args, **kwargs)

  def constraints_are_satisfied(self, nn, return_violation=False):
    """ Checks if the neural network nn satisfies the constraints. If return_violation
        is True, it returns a string representing the violation. """
    violation = ''
    if not self._check_leq_constraint(len(nn.layer_labels), self.max_num_layers):
      violation = 'too_many_layers'
    elif not self._check_geq_constraint(len(nn.layer_labels), self.min_num_layers):
      violation = 'too_few_layers'
    elif not self._check_leq_constraint(nn.get_total_mass(), self.max_mass):
      violation = 'too_much_mass'
    elif not self._check_geq_constraint(nn.get_total_mass(), self.min_mass):
      violation = 'too_little_mass'
    elif not self._check_leq_constraint(nn.get_out_degrees().max(), self.max_out_degree):
      violation = 'large_max_out_degree'
    elif not self._check_leq_constraint(nn.get_in_degrees().max(), self.max_in_degree):
      violation = 'large_max_in_degree'
    elif not self._check_leq_constraint(nn.conn_mat.sum(), self.max_num_edges):
      violation = 'too_many_edges'
    elif not self._check_leq_constraint(
                              self._finite_max_or_min(nn.num_units_in_each_layer, 1),
                              self.max_num_units_per_layer):
      violation = 'max_units_per_layer_exceeded'
    elif not self._check_geq_constraint(
                              self._finite_max_or_min(nn.num_units_in_each_layer, 0),
                              self.min_num_units_per_layer):
      violation = 'min_units_per_layer_not_exceeded'
    else:
      violation = self._child_constraints_are_satisfied(nn)
    return violation if return_violation else (violation == '')

  @classmethod
  def _check_leq_constraint(cls, value, bound):
    """ Returns true if bound is None or if value is less than or equal to bound. """
    return bound is None or (value <= bound)

  @classmethod
  def _check_geq_constraint(cls, value, bound):
    """ Returns true if bound is None or if value is greater than or equal to bound. """
    return bound is None or (value >= bound)

  def _child_constraints_are_satisfied(self, nn):
    """ Checks if the constraints of the child class are satisfied. """
    raise NotImplementedError('Implement in a child class.')

  @classmethod
  def _finite_max_or_min(cls, iterable, is_max):
    """ Returns the max ignorning Nones, nans and infs. """
    finite_vals = [x for x in iterable if x is not None and np.isfinite(x)]
    return max(finite_vals) if is_max else min(finite_vals)


class CNNConstraintChecker(NNConstraintChecker):
  """ A class for checking if a CNN satisfies constraints. """

  def __init__(self, max_num_layers, min_num_layers, max_mass, min_mass,
               max_in_degree, max_out_degree, max_num_edges,
               max_num_units_per_layer, min_num_units_per_layer,
               max_num_2strides=None):
    """ Constructor.
      max_num_2strides is the maximum number of 2-strides (either via pooling or conv
      operations) that the image can go through in the network.
    """
    super(CNNConstraintChecker, self).__init__(
      max_num_layers, min_num_layers, max_mass, min_mass,
      max_in_degree, max_out_degree, max_num_edges,
      max_num_units_per_layer, min_num_units_per_layer)
    self.max_num_2strides = max_num_2strides

  def _child_constraints_are_satisfied(self, nn):
    """ Checks if the constraints of the child class are satisfied. """
    img_inv_sizes = [piis for piis in nn.post_img_inv_sizes if piis != 'x']
    max_post_img_inv_sizes = None if self.max_num_2strides is None \
                                  else 2**self.max_num_2strides
    violation = ''
    if not self._check_leq_constraint(max(img_inv_sizes), max_post_img_inv_sizes):
      violation = 'too_many_2strides'
    return violation


class MLPConstraintChecker(NNConstraintChecker):
  """ A class for checking if a MLP satisfies constraints. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(MLPConstraintChecker, self).__init__(*args, **kwargs)

  def _child_constraints_are_satisfied(self, nn):
    """ Checks if the constraints of the child class are satisfied. """
    return ''


# An API to return an NN Domain using the constraints -----------------------
def get_nn_domain_from_constraints(nn_type, *args, **kwargs):
  """ nn_type is the type of the network.
      See CNNConstraintChecker, MLPConstraintChecker, NNConstraintChecker constructors
      for args and kwargs.
  """
  if nn_type[:3] == 'cnn':
    constraint_checker = CNNConstraintChecker(*args, **kwargs)
  elif nn_type[:3] == 'mlp':
    constraint_checker = MLPConstraintChecker(*args, **kwargs)
  else:
    raise ValueError('Unknown nn_type.')
  return NNDomain(nn_type, constraint_checker)

