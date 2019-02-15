"""
  A module for computing distances and kernels between neural networks.
  --kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

import numpy as np
# Local imports
from nas.nasbot.gp.kernel import ExpSumOfDistsKernel, SumOfExpSumOfDistsKernel
from nas.nasbot.nn import neural_network
from nas.nasbot.utils.oper_utils import opt_transport

DFLT_TRANSPORT_DIST = 'lp'
DFLT_CONN_COST_FUNC = 'linear'
DFLT_KERN_DIST_POWERS = 1
REPLACE_COST_INF_WITH = 7.65432e5
CONV_RES_RAW_COST_FRAC = 0.9

CNN_STRUCTURAL_PENALTY_GROUPS = ['all', 'conv', 'pool', 'fc']
MLP_STRUCTURAL_PENALTY_GROUPS = ['all', 'rectifier', 'sigmoid']
PATH_LENGTH_TYPES = ['shortest', 'longest', 'rw']


def _get_conv_filter_size_cost(labi, labj, conv_scale):
  """ Returns the cost for comparing two different convolutional filters. """
  conv_diff = float(abs(int(labi[-1]) - int(labj[-1])))
  return conv_scale * np.sqrt(conv_diff)


def get_cnn_layer_label_mismatch_penalties(non_assignment_penalty, max_conv_size=7,
                                           conv_scale=None):
  """ Gets the label mismatch matrix for a CNN. """
  conv_scale = np.sqrt(2)/10.0 if conv_scale is None else conv_scale
  cnn_layer_labels = neural_network.get_cnn_layer_labels(max_conv_size)
  num_labels = len(cnn_layer_labels)
  label_penalties = np.zeros((num_labels, num_labels))
  for i in range(num_labels):
    for j in range(i, num_labels):
      labi = cnn_layer_labels[i]
      labj = cnn_layer_labels[j]
      if labi == labj:
        cost = 0.0
      elif (labi.startswith('conv') and labj.startswith('conv')) or \
           (labi.startswith('res') and labj.startswith('res')):
        cost = _get_conv_filter_size_cost(labi, labj, conv_scale)
      elif (labi.startswith('conv') and labj.startswith('res')) or \
           (labi.startswith('res') and labj.startswith('conv')):
        raw_cost = _get_conv_filter_size_cost(labi, labj, conv_scale)
        cost = raw_cost if raw_cost > non_assignment_penalty else \
               (CONV_RES_RAW_COST_FRAC * raw_cost +
                (1-CONV_RES_RAW_COST_FRAC) * non_assignment_penalty)
        # When mapping the a convolutional block to a residual block, set the cost
        # to be in-between the cost for a conv-conv layer and the non_assignment_penalty.
      elif labi.endswith('pool') and labj.endswith('pool'):
        cost = 0.5
      else:
        cost = np.inf
      label_penalties[i, j] = cost * non_assignment_penalty
      label_penalties[j, i] = cost * non_assignment_penalty
  return cnn_layer_labels, label_penalties


def get_mlp_layer_label_mismatch_penalties(non_assignment_penalty, class_or_reg,
                                           list_of_activations=None):
  """ Gets the label mismatch penalty for an MLP. """
  rectifiers = neural_network.MLP_RECTIFIERS
  sigmoids = neural_network.MLP_SIGMOIDS
  non_linear_activations = rectifiers + sigmoids
  mlp_layer_labels = neural_network.get_mlp_layer_labels(class_or_reg,
                                                         list_of_activations)
  num_labels = len(mlp_layer_labels)
  label_penalties = np.zeros((num_labels, num_labels))
  for i in range(num_labels):
    for j in range(i, num_labels):
      labi = mlp_layer_labels[i]
      labj = mlp_layer_labels[j]
      if labi == labj:
        cost = 0.0
      elif (labi in rectifiers and labj in rectifiers) or \
           (labi in sigmoids and labj in sigmoids):
        cost = 0.1
      elif labi in non_linear_activations and labj in non_linear_activations:
        cost = 0.25
      else:
        cost = np.inf
      label_penalties[i, j] = cost * non_assignment_penalty
      label_penalties[j, i] = cost * non_assignment_penalty
  return mlp_layer_labels, label_penalties


class NNDistanceComputer(object):
  """ An abstract class for computing distances between neural networks. """

  def __init__(self):
    """ Constructor. """
    super(NNDistanceComputer, self).__init__()

  def __call__(self, X1, X2, *args, **kwargs):
    """ Evaluates the distances by calling evaluate. """
    return self.evaluate(X1, X2, *args, **kwargs)

  def evaluate(self, X1, X2, *args, **kwargs):
    """ Evaluates the distances between X1 and X2 and returns an n1 x n2 distance matrix.
        If X1 and X2 are single neural networks, returns a scalar. """
    if isinstance(X1, neural_network.NeuralNetwork) and \
       isinstance(X2, neural_network.NeuralNetwork):
      return self.evaluate_single(X1, X2, *args, **kwargs)
    else:
      n1 = len(X1)
      n2 = len(X2)
      X2 = X2 if X2 is not None else X1
      x1_is_x2 = X1 is X2

      all_ret = None
      es_is_iterable = None
      for i, x1 in enumerate(X1):
        X2_idxs = range(i, n2) if x1_is_x2 else range(n2)
        for j in X2_idxs:
          x2 = X2[j]
          # Compute the distances
          curr_ret = self.evaluate_single(x1, x2, *args, **kwargs)
          all_ret, es_is_iterable = self._add_to_all_ret(curr_ret, i, j, n1, n2,
                                                         all_ret, es_is_iterable)
          # Check if we need to do j and i as well.
          if x1_is_x2:
            all_ret, es_is_iterable = self._add_to_all_ret(curr_ret, j, i, n1, n2,
                                                           all_ret, es_is_iterable)
      return all_ret

  @classmethod
  def _add_to_all_ret(cls, curr_ret, i, j, n1, n2, all_ret=None, es_is_iterable=None):
    """ Adds the current result to all results. """
    if all_ret is None:
      if hasattr(curr_ret, '__iter__'):
        es_is_iterable = True
        all_ret = [np.zeros((n1, n2)) for _ in range(len(curr_ret))]
      else:
        es_is_iterable = False
        all_ret = np.zeros((n1, n2))
    if es_is_iterable:
      for k in range(len(curr_ret)):
        all_ret[k][i, j] = curr_ret[k]
    else:
      all_ret[i, j] = curr_ret
    return all_ret, es_is_iterable

  def evaluate_single(self, x1, x2, *args, **kwargs):
    """ Evaluates the distance between the two networks x1 and x2. """
    raise NotImplementedError('Implement in a child class.')


class OTMANNDistanceComputer(NNDistanceComputer):
  """ A distance between neural networks based on solving a transportation problem. """
  #pylint: disable=attribute-defined-outside-init

  def __init__(self, all_layer_labels, label_mismatch_penalty,
               non_assignment_penalty, structural_penalty_groups, path_length_types,
               dflt_mislabel_coeffs=None,
               dflt_struct_coeffs=None,
               connectivity_diff_cost_function=DFLT_CONN_COST_FUNC):
    """ Constructor. """
    super(OTMANNDistanceComputer, self).__init__()
    self.all_layer_labels = all_layer_labels
    self.label_mismatch_penalty = label_mismatch_penalty
    self.non_assignment_penalty = non_assignment_penalty
    self.structural_penalty_groups = structural_penalty_groups
    self.path_length_types = path_length_types
    self.all_path_length_categories = [x + '-' + y for x in self.structural_penalty_groups
                                       for y in self.path_length_types]
    self.dflt_mislabel_coeffs = dflt_mislabel_coeffs
    self.dflt_struct_coeffs = dflt_struct_coeffs
    self._set_up_connectivity_cost_function(connectivity_diff_cost_function)

  def _set_up_connectivity_cost_function(self, connectivity_diff_cost_function):
    """ Sets up the connectivity difference cost function. """
    self.connectivity_diff_cost_function = connectivity_diff_cost_function
    if connectivity_diff_cost_function == 'linear':
      self._conn_diff_cost_func = np.abs
    elif connectivity_diff_cost_function == 'sqrt':
      self._conn_diff_cost_func = lambda x: np.sqrt(np.abs(x))
    elif connectivity_diff_cost_function == 'log':
      self._conn_diff_cost_func = lambda x: np.log(np.abs(x))
    elif connectivity_diff_cost_function.startswith('poly'):
      poly_order = float(connectivity_diff_cost_function[4:])
      self._conn_diff_cost_func = lambda x: np.abs(x)**poly_order
    return

  def get_mislabel_cost_matrix(self, x1, x2):
    """ Returns the layer cost matrix from the graphs and label_mismatch_penalty """
    label_idxs_1 = [self.all_layer_labels.index(elem) for elem in x1.layer_labels]
    label_idxs_2 = [self.all_layer_labels.index(elem) for elem in x2.layer_labels]
    return self.label_mismatch_penalty[np.ix_(label_idxs_1, label_idxs_2)]

  def _get_cost_matrix_for_fwd_or_bkwd(self, x1_dists, x2_dists):
    """ Gets the cost matrix for one set of distances. """
    curr_cost_accumulation = np.zeros((x1_dists.shape[0], x2_dists.shape[0]))
    for dim in range(x1_dists.shape[1]):
      curr_diffs = x1_dists[:, dim][:, np.newaxis] - x2_dists[:, dim]
      curr_dim_costs = self._conn_diff_cost_func(curr_diffs)
      curr_cost_accumulation += curr_dim_costs
    curr_cost_matrix = curr_cost_accumulation/float(x1_dists.shape[1])
    return curr_cost_matrix

  def get_struct_cost_matrix(self, x1, x2):
    """ Gets a connectivity cost matrix. """
    x1_bkwd_ip_dists, x1_fwd_op_dists = x1.get_bkwd_ip_fwd_op_dists_of_all_layers(
                                          self.all_path_length_categories)
    x2_bkwd_ip_dists, x2_fwd_op_dists = x2.get_bkwd_ip_fwd_op_dists_of_all_layers(
                                          self.all_path_length_categories)
    bkwd_costs = self._get_cost_matrix_for_fwd_or_bkwd(x1_bkwd_ip_dists, x2_bkwd_ip_dists)
    fwd_costs = self._get_cost_matrix_for_fwd_or_bkwd(x1_fwd_op_dists, x2_fwd_op_dists)
    return (bkwd_costs + fwd_costs)/2

  @classmethod
  def get_ot_cost_matrix(cls, mislabel_cost_matrix, struct_cost_matrix,
                              mislabel_coeff, struct_coeff, non_assignment_penalty,
                              replace_cost_inf_with=REPLACE_COST_INF_WITH):
    """ Adds the two matrices and adds an additional dummy layer at he end of the rows and
        columns. Also makes some synthetic changes to enable OT computation. """
    # Add dummy layer
    cost_matrix = mislabel_coeff * mislabel_cost_matrix + \
                  struct_coeff * struct_cost_matrix
    row_add = non_assignment_penalty * np.ones((1, cost_matrix.shape[1]))
    col_add = non_assignment_penalty * np.ones((cost_matrix.shape[0] + 1, 1))
    col_add[-1] = 0.0
    cost_matrix = np.vstack((cost_matrix, row_add))
    cost_matrix = np.hstack((cost_matrix, col_add))
    # Replace infinities with a large value.
    if replace_cost_inf_with is not None and np.isfinite(replace_cost_inf_with):
      cost_matrix[np.logical_not(np.isfinite(cost_matrix))] = replace_cost_inf_with
    return cost_matrix

  def evaluate_single(self, x1, x2, mislabel_coeffs=None, struct_coeffs=None,
                      dist_type=DFLT_TRANSPORT_DIST):
    """ Evaluates the distances between two networks x1 and x2. dist_type is a
        string with options 'lp' or 'emd'.
    """
    # pylint: disable=arguments-differ
    #  Preprocessing
    mislabel_coeffs = mislabel_coeffs if mislabel_coeffs is not None else \
                      self.dflt_mislabel_coeffs
    struct_coeffs = struct_coeffs if struct_coeffs is not None else \
                    self.dflt_struct_coeffs
    if not hasattr(mislabel_coeffs, '__len__'):
      mislabel_coeffs = [mislabel_coeffs]
    if not hasattr(struct_coeffs, '__len__'):
      struct_coeffs = [struct_coeffs]
    assert len(mislabel_coeffs) == len(struct_coeffs)
    # Compute the types of distances we need to compute.
    types_of_distances = dist_type.split('-')
    # Create data for the transportation problem.
    total_wt_1 = sum(x1.layer_masses)
    total_wt_2 = sum(x2.layer_masses)
    supplies = np.append(x1.layer_masses, total_wt_2)
    demands = np.append(x2.layer_masses, total_wt_1)
    # Get the mislabel and structural cost matrices
    mislabel_cost_matrix = self.get_mislabel_cost_matrix(x1, x2)
    struct_cost_matrix = self.get_struct_cost_matrix(x1, x2)
    # Go through each coefficient and repeat
    ret = []
    for coeff_idx in range(len(mislabel_coeffs)):
      curr_ot_cost_matrix = self.get_ot_cost_matrix(
                              mislabel_cost_matrix, struct_cost_matrix,
                              mislabel_coeffs[coeff_idx], struct_coeffs[coeff_idx],
                              self.non_assignment_penalty)
      _, min_val, emd = opt_transport(supplies, demands, curr_ot_cost_matrix)
      # Below, emd and lp-norm-by-max (or min, sum) are not distances.
      for dt in types_of_distances:
        if dt == 'lp':
          ret.append(min_val)
        elif dt == 'emd':
          ret.append(emd)
        elif dt == 'lp_norm_by_max':
          dist_lp_norm = min_val / max(total_wt_1, total_wt_2)
          ret.append(dist_lp_norm)
        elif dt == 'log_lp':
          ret.append(np.log(1 + min_val))
        else:
          raise ValueError('Unknown dist_type \'%s\'.'%(dist_type))
    return ret


class DistProdNNKernel(ExpSumOfDistsKernel):
  """ Computes a kernel using the transportation distance as the distance. """

  def __init__(self, trans_dist_computer, mislabel_coeffs, struct_coeffs, betas, scale,
               powers=DFLT_KERN_DIST_POWERS, dist_type=DFLT_TRANSPORT_DIST):
    """ Constructor. """
    self.trans_dist_computer = trans_dist_computer
    self.dist_type = dist_type
    self.num_dists = len(betas)
    self.num_dist_types = len(self.dist_type.split('-'))
    self.struct_coeffs = struct_coeffs \
      if hasattr(struct_coeffs, '__iter__') else [struct_coeffs]
    self.mislabel_coeffs = mislabel_coeffs \
      if hasattr(mislabel_coeffs, '__iter__') else [mislabel_coeffs]
    if self.num_dists != self.num_dist_types * len(self.struct_coeffs):
      raise ValueError(('The number of beta values(%d) should be %d times that of the' +
        ' number of struct_coeffs(%d) for dist_type=%s.')%(len(betas),
        self.num_dist_types, self.dist_type))
    # Call super constructor
    super(DistProdNNKernel, self).__init__(self.compute_dists, betas, scale,
                                           powers, self.num_dists)

  def compute_dists(self, X1, X2):
    """ Given two lists of neural networks computes the distance between the two. """
    return self.trans_dist_computer(X1, X2, self.mislabel_coeffs, self.struct_coeffs,
                                    self.dist_type)


class DistSumNNKernel(SumOfExpSumOfDistsKernel):
  """ Computes a kernel using the transportation distance as the distance. """

  def __init__(self, trans_dist_computer, mislabel_coeffs, struct_coeffs, alphas,
               betas, powers=DFLT_KERN_DIST_POWERS, dist_type=DFLT_TRANSPORT_DIST):
    """ Constructor. """
    self.trans_dist_computer = trans_dist_computer
    self.dist_type = dist_type
    self.num_dists = len(betas)
    self.num_dist_types = len(self.dist_type.split('-'))
    self.struct_coeffs = struct_coeffs \
      if hasattr(struct_coeffs, '__iter__') else [struct_coeffs]
    self.mislabel_coeffs = mislabel_coeffs \
      if hasattr(mislabel_coeffs, '__iter__') else [mislabel_coeffs]
    if self.num_dists != self.num_dist_types * len(self.struct_coeffs):
      raise ValueError(('The number of beta values(%d) should be %d times that of the' +
        ' number of struct_coeffs(%d) for dist_type=%s.')%(len(betas),
        self.num_dist_types, self.dist_type))
    groups = self._get_groups(self.num_dists, self.num_dist_types)
    # Call super constructor
    super(DistSumNNKernel, self).__init__(self.compute_dists, alphas, groups, betas,
                                          powers)

  @classmethod
  def _get_groups(cls, num_dists, num_dist_types):
    """ Returns the number of groups. """
    num_dists_per_group = num_dists/num_dist_types
    groups = [[(i + num_dist_types * j) for j in range(num_dists_per_group)]
              for i in range(num_dist_types)]
    return groups

  def compute_dists(self, X1, X2):
    """ Given two lists of neural networks computes the distance between the two. """
    return self.trans_dist_computer(X1, X2, self.mislabel_coeffs, self.struct_coeffs,
                                    self.dist_type)

# APIs to return a distance computer or kernel -------------------------------------------
def get_otmann_distance_from_args(nn_type, non_assignment_penalty,
                                     connectivity_diff_cost_function=DFLT_CONN_COST_FUNC):
  """ Obtains a transport distance computer from dists. """
  if nn_type.startswith('cnn'):
    all_layer_labels, label_mismatch_penalty = \
      get_cnn_layer_label_mismatch_penalties(non_assignment_penalty)
    struct_penalty_groups = CNN_STRUCTURAL_PENALTY_GROUPS
  elif nn_type.startswith('mlp'):
    all_layer_labels, label_mismatch_penalty = \
      get_mlp_layer_label_mismatch_penalties(non_assignment_penalty,
                    nn_type[4:])
    struct_penalty_groups = MLP_STRUCTURAL_PENALTY_GROUPS
  # Now create a tp_comp object
  tp_comp = OTMANNDistanceComputer(all_layer_labels,
              label_mismatch_penalty, non_assignment_penalty,
              struct_penalty_groups, PATH_LENGTH_TYPES,
              connectivity_diff_cost_function=connectivity_diff_cost_function)
  return tp_comp

def get_default_otmann_distance(nn_type, non_assignment_penalty):
  """ The otmann distance with default parameters. """
  return get_otmann_distance_from_args(nn_type, non_assignment_penalty)

def generate_otmann_kernel_from_params(
  kernel_type, # The kernel type, should be 'sum' or 'product'
  all_layer_labels, label_mismatch_penalty, # mandatory distance args
  non_assignment_penalty, structural_penalty_groups, path_length_types,
  mislabel_coeffs, struct_coeffs, betas, scales, # mandatory kernel args
  connectivity_diff_cost_function=DFLT_CONN_COST_FUNC, # optional dist args
  powers=DFLT_KERN_DIST_POWERS, dist_type=DFLT_TRANSPORT_DIST): # optional kernel args
  #pylint: disable=too-many-arguments
  """ Generates a OTMANNKernel from all parameters for the distance computer
      and the kernel.
      scales should be a scalar if kernel_type is 'prod' and a list if 'sum'.
  """
  tp_comp = OTMANNDistanceComputer(all_layer_labels, label_mismatch_penalty,
              non_assignment_penalty, structural_penalty_groups, path_length_types,
              connectivity_diff_cost_function=connectivity_diff_cost_function)
  if kernel_type == 'prod':
    return DistProdNNKernel(tp_comp, mislabel_coeffs, struct_coeffs,
                            betas, scales, powers, dist_type)
  elif kernel_type == 'sum':
    return DistSumNNKernel(tp_comp, mislabel_coeffs, struct_coeffs, scales,
                           betas, powers, dist_type)

