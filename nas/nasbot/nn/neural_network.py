"""
  Harness for storing and computing various representations for neural networks.
  --kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used

from copy import deepcopy
import numpy as np
# Local imports
from ..utils import graph_utils
from ..utils.general_utils import get_nonzero_indices_in_vector, reorder_list_or_array, \
                                reorder_rows_and_cols_in_matrix

# global constants
STORE_GRAPH_PARAMS_BY_DEFAULT = False
SOFTMAX_LINEAR_LAYER_MASS = 0
POOL_LAYER_MASS = 0
UNIV_MLP_RECTIFIERS = ['relu', 'relu6', 'crelu', 'relu-x', 'leaky-relu', 'softplus',
                       'elu']
UNIV_MLP_SIGMOIDS = ['logistic', 'tanh', 'step']
MLP_RECTIFIERS = ['relu', 'crelu', 'leaky-relu', 'softplus', 'elu']
MLP_SIGMOIDS = ['logistic', 'tanh']
_NON_PROC_LAYERY_MASS_FRAC = 0.1
_FC_LAYER_MASS_COEFF = 0.1 # coefficient for the fully connected layers


# Some utilities we will need below ---------------------------------------------
def is_a_pooling_layer_label(layer_label):
  """ Returns true if a pooling layer. """
  return 'pool' in layer_label

def is_a_conv_layer_label(layer_label):
  """ Returns true if a convolutional layer. """
  return 'conv' in layer_label or 'res' in layer_label


# Exceptions we will define that can be handled later on ------------------------
class CNNImageSizeMismatchException(Exception):
  """ Exception for mismatches in image sizes. """
  def __init__(self, descr):
    """ Constructor. """
    super(CNNImageSizeMismatchException, self).__init__()
    self.descr = descr
  def __str__(self):
    """ Returns string representation. """
    return 'CNNImageSizeMismatchException: %s'%(self.descr)

class CNNNoConvAfterIPException(Exception):
  """ Exception if the layers immediately after the input are not convloutions. """
  def __init__(self, descr):
    """ Constructor. """
    super(CNNNoConvAfterIPException, self).__init__()
    self.descr = descr
  def __str__(self):
    """ Returns string representation. """
    return 'CNNNoConvAfterIPException: %s'%(self.descr)


# Neural network class ----------------------------------------------------------
class NeuralNetwork(object):
  """ Base class for all neural networks. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, nn_class, layer_labels, conn_mat, num_units_in_each_layer,
               all_layer_label_classes, layer_label_similarities=None):
    """ nn_class is a string describing whether it is a FF network, CNN etc.
      Parameters specific to the NN
      layer labels: is a list of labels for each layer.
      conn_mat: is the connectivity matrix. If (i,j) is 1 it means there is a
                connection between layer i and j.
      Global parameters
      all_layer_label_classes: a list of all labels a network can take.
      layer_label_similarities: measures of similarity for each label type.
    """
    self.nn_class = nn_class
    self.layer_labels = layer_labels
    self.conn_mat = conn_mat
    self.num_units_in_each_layer = np.array(num_units_in_each_layer)
    if not hasattr(self, 'mandatory_child_attributes'):
      self.mandatory_child_attributes = []
    # Global parameters
    self.all_layer_label_classes = all_layer_label_classes
    self.layer_label_similarities = layer_label_similarities
    self._set_up()

  def _set_up(self):
    """ Additional constructor actions. """
    # Prelims
    self.num_layers = len(self.layer_labels)
    self.bkwd_ip_dists_of_layers = None
    self.fwd_op_dists_of_layers = None
    # First do a topological sort
    self._topological_sort()
    # Compute layer masses
    self._compute_layer_masses()
    # Check for other parameters
    self.num_internal_layers = len(self.layer_labels) - 2
    self.num_processing_layers = len([ll for ll in self.layer_labels if ll not in
                                      ['ip', 'op', 'softmax', 'linear']])
    self.internal_layer_idxs = np.delete(range(self.num_layers), [self.get_ip_layer_idx(),
                                 self.get_op_layer_idx()])
    self.internal_layer_masses = self.layer_masses[self.internal_layer_idxs]
    # For storing additional matrices we might need along the way
    # Compute the distances by short circuiting each type of layer.
    self._bkwd_ip_fwd_op_dist_type_order = ['all'] + self._get_child_layer_groups()
    self._path_length_type_order = ['shortest', 'longest', 'rw']
    self._all_path_length_categories = [x + '-' + y
                                        for x in self._bkwd_ip_fwd_op_dist_type_order
                                        for y in self._path_length_type_order]
    self._compute_ip_op_path_lengths()
    # Now re-organise the above data into a matrix of num-layer x num-layer-type matrix.
    self.bkwd_ip_dists_of_layers, self.fwd_op_dists_of_layers = \
      self.get_bkwd_ip_fwd_op_dists_of_all_layers(self._all_path_length_categories)
    # Finally, check if this is a valid network
    assert self._check_if_valid_network()

  def _topological_sort(self):
    """ Reorders the layers so that they are in a topologically sorted fashion. """
    top_order, has_cycles = graph_utils.kahn_topological_sort(self.conn_mat,
                              self.layer_labels.index('ip'))
    assert not has_cycles
    self.layer_labels = reorder_list_or_array(self.layer_labels, top_order)
    self.num_units_in_each_layer = reorder_list_or_array(self.num_units_in_each_layer,
                                                      top_order)
    self.conn_mat = reorder_rows_and_cols_in_matrix(self.conn_mat, top_order)
    self._child_attrs_topological_sort(top_order)

  def _compute_layer_masses(self):
    """ Computes the layer masses for all layers. """
    self._child_compute_layer_masses()

  def _child_compute_layer_masses(self):
    """ Computes the layer masses for all layers. """
    raise NotImplementedError('Implement _child_compute_layer_masses in a child class.')

  def get_children(self, layer_idx):
    """ Returns the children of layer_idx. """
    return graph_utils.get_children(layer_idx, self.conn_mat)

  def get_parents(self, layer_idx):
    """ Returns the parents of layer_idx. """
    return graph_utils.get_parents(layer_idx, self.conn_mat)

  def get_edges(self):
    """ Returns the edges of network as a list. """
    return self.conn_mat.keys()

  def get_total_num_edges(self):
    """ Returns the total number of edges. """
    return self.conn_mat.sum()

  def _get_layer_indices_of_layer_type(self, layer_type, layer_labels=None):
    """ Gets the layer indices of layer_type. """
    return self._get_layer_indices_of_layer_or_group_type(layer_type, 'layer',
                                                          layer_labels)

  def _get_layer_indices_of_group_type(self, group_type, layer_labels=None):
    """ Gets the layer indices of layers belonging to that group. """
    return self._get_layer_indices_of_layer_or_group_type(group_type, 'group',
                                                          layer_labels)

  def _get_layer_indices_of_layer_or_group_type(self, label, layer_or_group=None,
                                                layer_labels=None):
    """ Gets the layer indices of layers belonging to the group 'label' if layer_or_group
        is 'group'. Else it returns the indices where the layer_label is label. """
    layer_labels = layer_labels if layer_labels is not None else self.layer_labels
    if label == 'all':
      return list(range(len(layer_labels)))
    # Otherwise first determine whether it is a group or a layer
    if layer_or_group == None:
      group_label = self._get_layer_group_for_layer_label(label)
      layer_or_group = 'group' if group_label is None else 'layer'
    # Now return accordingly
    if layer_or_group == 'layer':
      return [i for i, x in enumerate(layer_labels) if x == label]
    elif layer_or_group == 'group':
      return [i for i, x in enumerate(layer_labels) if
              self._get_layer_group_for_layer_label(x) == label]
    else:
      raise ValueError('layer_or_group should be either \'layer\' or \'group\' or None.')

  def get_ip_layer_idx(self):
    """ Returns the index of the input layer. """
    return self.layer_labels.index('ip')

  def get_op_layer_idx(self):
    """ Returns the index of the output layer. """
    return self.layer_labels.index('op')

  def get_total_mass(self):
    """ Returns the total mass of the network. """
    return self.layer_masses.sum()

  def get_in_degrees(self):
    """ Returns the in degrees of all nodes. """
    return np.array(self.conn_mat.sum(axis=0)).ravel()

  def get_out_degrees(self):
    """ Returns the out degrees of all nodes. """
    return np.array(self.conn_mat.sum(axis=1)).ravel()

  def get_distances_from_ip(self, dist_type='all-shortest'):
    """ Returns the distances from the input. """
    return self._bkwd_dists_to_ip[dist_type]

  def get_distances_to_op(self, dist_type='all-shortest'):
    """ Returns the distances from the input. """
    return self._fwd_dists_to_op[dist_type]

  def _check_if_valid_network(self):
    """ Returns true if the network is a valid network. """
    # Check if the lenghts of the labels and node masss are the same
    assert len(self.layer_labels) == len(self.num_units_in_each_layer)
    # If there are no processing layers, there need not be more than 3 layers (ip, op and
    # decision).
    assert self.num_processing_layers > 0 or self.num_layers == 3
    # Check if the labels are correct
    input_layers = self._get_layer_indices_of_layer_type('ip')
    output_layers = self._get_layer_indices_of_layer_type('op')
    assert len(input_layers) == 1
    assert len(output_layers) == 1
    # A bunch of checks based on the APSP distances
    # Check if the forward distances to 'ip' and backward distances to 'op' are 0.
    ip_idx = self.get_ip_layer_idx()
    op_idx = self.get_op_layer_idx()
    # Check if self distances are all 0
    assert np.all(self._fwd_dists_to_op['all-rw'][op_idx] == 0)
    assert np.all(self._bkwd_dists_to_ip['all-rw'][ip_idx] == 0)
    # Check to see if the forward distances from the ip and backward distances from the
    # output are finite.
    assert np.all(np.isfinite(self._fwd_dists_to_op['all-longest']))
    assert np.all(np.isfinite(self._bkwd_dists_to_ip['all-longest']))
    # Check if all labels are valid
    assert all([ll in self.all_layer_label_classes for ll in self.layer_labels])
    # Finally check if there are any conditions that the child network needs to satisfy.
    assert self._child_check_if_valid_network()
    return True

  def _child_check_if_valid_network(self):
    """ Used for any tests the child class might want to check. """
    raise NotImplementedError('Impelement in a child class.')

  def _child_attrs_topological_sort(self, top_order):
    """ Topologically sort any attributes of the child class. """
    raise NotImplementedError('Impelement in a child class.')

  def get_layer_descr(self, layer_idx, *_):
    """ Returns a string describing the layer. Used in visualing the layer. """
    if isinstance(self.num_units_in_each_layer[layer_idx], (int, float, long)) and \
      np.isfinite(self.num_units_in_each_layer[layer_idx]):
      num_units_descr = str(int(self.num_units_in_each_layer[layer_idx])) + ','
    else:
      num_units_descr = ''
    return '#%d %s, %s\n(%d)'%(layer_idx, self.layer_labels[layer_idx],
                             num_units_descr, self.layer_masses[layer_idx])
#     return '#%d %s, %s'%(layer_idx, self.layer_labels[layer_idx],
#                          num_units_descr) # cleantech slides

  def get_edge_weights_from_conn_mat(self):
    """ Returns a matrix of edge weights with infinities for non-edges. """
    # First compute the preliminary edge weights matrix.
    edge_weights = (deepcopy(self.conn_mat)).toarray()
    edge_weights[edge_weights == 0] = np.inf
    return edge_weights

  def get_layer_or_group_edge_weights_from_edge_weights(self, edge_weights,
                                                        layer_or_group_label):
    """ Gets the edge weights corresponding to the layer_or_group_label. """
    layer_or_group_edge_weights = deepcopy(edge_weights)
    layer_or_group_edge_weights[layer_or_group_edge_weights == 1] = 0
    curr_layers = self._get_layer_indices_of_layer_or_group_type(layer_or_group_label)
    for cl in curr_layers:
      layer_or_group_edge_weights[cl, :] = edge_weights[cl, :]
    return layer_or_group_edge_weights

  def _compute_apsps(self):
    """ Computes various kinds of all pairs shortest paths. """
    edge_weights = self.get_edge_weights_from_conn_mat()
    self._fwd_apsps = {}
    for lg in self._bkwd_ip_fwd_op_dist_type_order:
      curr_edge_weights = self.get_layer_or_group_edge_weights_from_edge_weights(
                            edge_weights, lg)
      self._fwd_apsps[lg] = graph_utils.apsp_floyd_warshall_costs(curr_edge_weights)

  def _compute_ip_op_path_lengths(self):
    """ Computes the various kinds of shortest path lengths to ip and op. """
    edge_weights = self.get_edge_weights_from_conn_mat()
    self._fwd_dists_to_op = {}
    self._bkwd_dists_to_ip = {}
    for lg in self._bkwd_ip_fwd_op_dist_type_order:
      curr_edge_weights = self.get_layer_or_group_edge_weights_from_edge_weights(
                            edge_weights, lg)
      curr_edge_weights_T = self.get_layer_or_group_edge_weights_from_edge_weights(
                              edge_weights.T, lg)
      for plt in self._path_length_type_order:
        curr_key = lg + '-' + plt
        # compute_nn_path_lengths expects the forward graph and computes the distances
        # to the last node.
        self._bkwd_dists_to_ip[curr_key] = graph_utils.compute_nn_path_lengths(
          curr_edge_weights_T, list(reversed(range(self.num_layers))), plt)
        self._fwd_dists_to_op[curr_key] = graph_utils.compute_nn_path_lengths(
          curr_edge_weights, list((range(self.num_layers))), plt)

  def get_bkwd_ip_fwd_op_dists_of_layer(self, layer_idx, path_length_types):
    """ Returns the forward and backward distances of the layer in layer_idx. """
    bkwd_ip_dists = [self._bkwd_dists_to_ip[dist_type][layer_idx] for dist_type in
                     path_length_types]
    fwd_op_dists = [self._fwd_dists_to_op[dist_type][layer_idx] for dist_type in
                    path_length_types]
    return bkwd_ip_dists, fwd_op_dists

  def get_bkwd_ip_fwd_op_dists_of_all_layers(self, path_length_categories):
    """ Returns the forward and backward distances of all layers except the input/output
        layers. """
    if self.bkwd_ip_dists_of_layers is not None:
      col_reordering = [self._all_path_length_categories.index(elem)
                        for elem in path_length_categories]
      return (self.bkwd_ip_dists_of_layers[:, col_reordering],
              self.fwd_op_dists_of_layers[:, col_reordering])
    else:
      bkwd_ip_dists_of_layers = []
      fwd_op_dists_of_layers = []
      for lidx in range(self.num_layers):
        curr_bip, curr_fop = self.get_bkwd_ip_fwd_op_dists_of_layer(lidx,
                               path_length_categories)
        bkwd_ip_dists_of_layers.append(curr_bip)
        fwd_op_dists_of_layers.append(curr_fop)
      return np.array(bkwd_ip_dists_of_layers), np.array(fwd_op_dists_of_layers)

  def _get_property_and_set_if_to_store(self, property_name, property_comp_function,
                                        to_store=STORE_GRAPH_PARAMS_BY_DEFAULT):
    """ A function for returning and setting a property if required. Will be used by many
        of the functions below. """
    curr_stored_val = getattr(self, property_name)
    if curr_stored_val is not None:
      return curr_stored_val
    else:
      computed_val = property_comp_function()
      if to_store:
        setattr(self, property_name, computed_val)
      return computed_val

  # Some abstract methods ============================================================
  @classmethod
  def _get_child_layer_groups(cls):
    """ Returns layer groups for the network. """
    raise NotImplementedError('Implement for a child class.')

  @classmethod
  def _get_layer_group_for_layer_label(cls, layer_label):
    """ Returns layer group for layer. """
    raise NotImplementedError('Implement for a child class.')



# Some specific instances of Neural network ============================================
def _check_if_layers_before_op_are(conn_mat, op_layer_idx, layer_labels, label_val):
  """ Checks if all layers before the output have label_val as the label. """
  layers_before_op = get_nonzero_indices_in_vector(conn_mat[:, op_layer_idx])
  for layer_idx in layers_before_op:
    if layer_labels[layer_idx] != label_val:
      return False
  return True

def compute_num_channels_at_each_layer(nn):
  """ Computes the number of channels at each layer. """
  num_channels_in_to_each_layer = [None]
  num_channels_out_of_each_layer = [1]
  for layer_idx in range(1, nn.num_layers-1):
    ll = nn.layer_labels[layer_idx]
    curr_parents = nn.get_parents(layer_idx)
    channels_from_parents = [num_channels_out_of_each_layer[par_idx] for
                             par_idx in curr_parents]
    # Determine the number of channels in
    curr_in = sum(channels_from_parents)
    # Determine the number of channels out
    if is_a_pooling_layer_label(ll):
      curr_out = curr_in
    elif ll in ['softmax', 'linear']:
      curr_out = None
    else:
      curr_out = nn.num_units_in_each_layer[layer_idx]
    num_channels_in_to_each_layer.append(curr_in)
    num_channels_out_of_each_layer.append(curr_out)
  # For the output layer
  num_channels_in_to_each_layer.append(None)
  num_channels_out_of_each_layer.append(None)
  return num_channels_in_to_each_layer, num_channels_out_of_each_layer

def compute_layer_masses(num_channels_in_to_each_layer, num_channels_out_of_each_layer,
                         layer_labels):
  """ Computes the layer masses for all layers. """
  num_layers = len(num_channels_in_to_each_layer)
  layer_masses = np.zeros((num_layers))
  num_decision_layers = 0
  for layer_idx in range(num_layers):
    if layer_labels[layer_idx] in ['softmax', 'linear']:
      num_decision_layers += 1
      continue # we will do this later.
    elif layer_labels[layer_idx] in ['ip', 'op']:
      continue # we will do this later.
    if is_a_pooling_layer_label(layer_labels[layer_idx]):
      layer_masses[layer_idx] = num_channels_in_to_each_layer[layer_idx]
    else:
      layer_masses[layer_idx] = \
        num_channels_in_to_each_layer[layer_idx] * \
        num_channels_out_of_each_layer[layer_idx]
      if layer_labels[layer_idx] == 'fc':
        layer_masses[layer_idx] *= _FC_LAYER_MASS_COEFF
  # Now add layer mass for the non-processing layers
  total_proc_layer_mass = layer_masses.sum()
  non_proc_layer_mass = max(_NON_PROC_LAYERY_MASS_FRAC * total_proc_layer_mass, 100)
  decision_layer_mass = non_proc_layer_mass / float(num_decision_layers)
  for layer_idx, ll in enumerate(layer_labels):
    if ll in ['softmax', 'linear']:
      layer_masses[layer_idx] = decision_layer_mass
    elif ll in ['ip', 'op']:
      layer_masses[layer_idx] = non_proc_layer_mass
  return layer_masses


class ConvNeuralNetwork(NeuralNetwork):
  """ Implements a convolutional neural network. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, layer_labels, conn_mat, num_filters_in_each_layer,
               strides, all_layer_label_classes=None, layer_label_similarities=None):
    """ Constructor. """
    self.mandatory_child_attributes = ['strides']
    self.strides = strides
    super(ConvNeuralNetwork, self).__init__('cnn', layer_labels, conn_mat,
          num_filters_in_each_layer, all_layer_label_classes,
          layer_label_similarities)

  def _child_compute_layer_masses(self):
    """ Computes the layer masses and the number of channels at each layer. """
    self.num_channels_in_to_each_layer, self.num_channels_out_of_each_layer = \
      compute_num_channels_at_each_layer(self)
    self.layer_masses = compute_layer_masses(self.num_channels_in_to_each_layer,
      self.num_channels_out_of_each_layer, self.layer_labels)

  @classmethod
  def _check_if_parent_image_sizes_are_valid(cls, image_sizes):
    """ Returns true if all image sizes are equal. """
    if len(image_sizes) == 0:
      return 'This node has no parents.'
    elif not image_sizes[1:] == image_sizes[:-1]:
      return 'Not all image sizes are equal: %s.'%(str(image_sizes))
    elif image_sizes[0] == None:
      return 'Image size None encountered: %s.'%(str(image_sizes))
    return ''

  def _check_image_size_consistency(self):
    """ Checks if the images sizes are consistent across all layers. """
    self.pre_img_inv_sizes = [None for _ in range(self.num_layers)]
    self.post_img_inv_sizes = [None for _ in range(self.num_layers)]
    self.pre_img_inv_sizes[0] = 1
    self.post_img_inv_sizes[0] = 1
    assert self.strides[0] is None
    for layer_idx in range(1, self.num_layers):
      # First check if the stride value is consistent
      if is_a_conv_layer_label(self.layer_labels[layer_idx]):
        assert self.strides[layer_idx] in [1, 2]
      else:
        assert self.strides[layer_idx] is None
      # Obtain the parents and check if the sizes are consistent
      curr_parents = self.get_parents(layer_idx)
      parent_post_img_sizes = [self.post_img_inv_sizes[x] for x in curr_parents]
      # Check that the sizes are valid if it is a convolutional or pooling layer.
      if is_a_pooling_layer_label(self.layer_labels[layer_idx]) or \
         is_a_conv_layer_label(self.layer_labels[layer_idx]):
        # Checking only the first parent because we are checking for equality later.
        assert isinstance(parent_post_img_sizes[0], (int, int, float)) and \
               parent_post_img_sizes[0] > 0
      # Check parent image sizes and if they are consistent
      check_parent_img_sizes = self._check_if_parent_image_sizes_are_valid(
                                 parent_post_img_sizes)
      if check_parent_img_sizes != '':
        raise CNNImageSizeMismatchException('layer %d (%s): %s.'%(layer_idx,
                          self.layer_labels[layer_idx], check_parent_img_sizes))
      # If so, now update the child's parameters
      self.pre_img_inv_sizes[layer_idx] = parent_post_img_sizes[0]
      if is_a_pooling_layer_label(self.layer_labels[layer_idx]) or \
          (is_a_conv_layer_label(self.layer_labels[layer_idx]) and
           self.strides[layer_idx] == 2):
        self.post_img_inv_sizes[layer_idx] = 2 * self.pre_img_inv_sizes[layer_idx]
      elif is_a_conv_layer_label(self.layer_labels[layer_idx]):
        self.post_img_inv_sizes[layer_idx] = self.pre_img_inv_sizes[layer_idx]
      elif self.layer_labels[layer_idx] in ['fc', 'softmax', 'op']:
        self.post_img_inv_sizes[layer_idx] = 'x'
    return True

  def _check_layer_units_and_masses(self):
    """ Checks the number of units and masses for the pool layers. """
    for idx, ll in enumerate(self.layer_labels):
      if is_a_conv_layer_label(ll):
        assert self.num_units_in_each_layer[idx] > 0
        assert self.layer_masses[idx] > 0
      elif is_a_pooling_layer_label(ll):
        pass
      elif ll in ['ip', 'op', 'softmax', 'linear']:
        assert self.num_units_in_each_layer[idx] is None
    return True

  def _check_layers_after_ip(self):
    """ Checks if layers after the input are all convolutional layers. """
    ip_ch_labels = [self.layer_labels[i] for i in self.get_children(0)]
    ip_ch_labels_are_conv = [is_a_conv_layer_label(lab) for lab in ip_ch_labels]
    if not all(ip_ch_labels_are_conv):
      raise CNNNoConvAfterIPException('Children of input layer are not convolutional:' +
        ' %s.'%(ip_ch_labels))
    return True

  def _child_check_if_valid_network(self):
    """ Runs any checks relevant for a CNN. """
    # Test if all layers before the output are softmax.
    assert _check_if_layers_before_op_are(self.conn_mat, self.get_op_layer_idx(),
                                          self.layer_labels, 'softmax')
    assert self._check_image_size_consistency()
    assert self._check_layer_units_and_masses()
    assert self._check_layers_after_ip()
    return True

  def _child_attrs_topological_sort(self, top_order):
    """ Topologically sort child attributes. """
    self.strides = reorder_list_or_array(self.strides, top_order)

  def get_layer_descr(self, layer_idx, for_pres=False):
    """ Returns a string description describing the layer. """
    #pylint: disable=arguments-differ
    if not for_pres:
      if is_a_conv_layer_label(self.layer_labels[layer_idx]) and \
         self.strides[layer_idx] == 2:
        stride_str = ', /2'
      else:
        stride_str = ''
      img_size_str = 'None' if self.pre_img_inv_sizes[layer_idx] is None else \
                     str(self.pre_img_inv_sizes[layer_idx])
      child_str = ' [%s%s]'%(img_size_str, stride_str)
      return super(ConvNeuralNetwork, self).get_layer_descr(layer_idx) + child_str
    else:
      if isinstance(self.num_units_in_each_layer[layer_idx], (int, float, long)) and \
        np.isfinite(self.num_units_in_each_layer[layer_idx]):
        num_units_descr = ', ' + str(int(self.num_units_in_each_layer[layer_idx]))
      elif is_a_pooling_layer_label(self.layer_labels[layer_idx]):
        num_units_descr = ', 1'
      else:
        num_units_descr = ''
      if self.strides[layer_idx] == 2:
        stride_str = ' /2'
      else:
        stride_str = ''
      return '%d: %s%s%s\n(%d)'%(layer_idx, self.layer_labels[layer_idx], stride_str,
                                 num_units_descr, self.layer_masses[layer_idx])

  @classmethod
  def _get_child_layer_groups(cls):
    """ Returns layer groups for a CNN. """
    return ['conv', 'pool', 'fc']

  @classmethod
  def _get_layer_group_for_layer_label(cls, layer_label):
    """ Returns layer group for layer. """
    if (layer_label.startswith('conv') and layer_label != 'conv') or \
       (layer_label.startswith('res') and layer_label != 'res'):
      return 'conv'
    elif layer_label.endswith('pool') and layer_label != 'pool':
      return 'pool'
    elif layer_label in ['fc', 'softmax']:
      return layer_label
    else:
      return None


class MultiLayerPerceptron(NeuralNetwork):
  """ Implements a multi-layer perceptron. """
  #pylint: disable=attribute-defined-outside-init

  def __init__(self, class_or_reg, layer_labels, conn_mat, num_units_in_each_layer,
               all_layer_label_classes=None, layer_label_similarities=None):
    """ Constructor. """
    self.mandatory_child_attributes = []
    self.class_or_reg = class_or_reg
    if class_or_reg.lower().startswith('reg'):
      nn_class = 'mlp-reg'
    elif class_or_reg.lower().startswith('class'):
      nn_class = 'mlp-class'
    else:
      raise ValueError('class_or_reg should be either class or reg. Given %s.'%(
                       class_or_reg))
    super(MultiLayerPerceptron, self).__init__(nn_class, layer_labels, conn_mat,
      num_units_in_each_layer, all_layer_label_classes,
      layer_label_similarities)

  def _child_check_if_valid_network(self):
    """ Runs any checks relevant for a MLP. """
    # Test if all layers before the output are linear/softmax.
    last_layer_label = 'linear' if self.nn_class == 'mlp-reg' else 'softmax'
    return _check_if_layers_before_op_are(self.conn_mat, self.get_op_layer_idx(),
                                          self.layer_labels, last_layer_label)

  def _child_attrs_topological_sort(self, top_order):
    """ Topologically sort child attributes. """
    pass

  def _child_compute_layer_masses(self):
    """ Computes the layer masses and the number of channels at each layer. """
    self.num_channels_in_to_each_layer, self.num_channels_out_of_each_layer = \
      compute_num_channels_at_each_layer(self)
    self.layer_masses = compute_layer_masses(self.num_channels_in_to_each_layer,
      self.num_channels_out_of_each_layer, self.layer_labels)

  @classmethod
  def _get_child_layer_groups(cls):
    """ Returns layer groups for an MLP. """
    return ['rectifier', 'sigmoid']

  @classmethod
  def _get_layer_group_for_layer_label(cls, layer_label):
    """ Returns layer group for layer. """
    if layer_label in MLP_RECTIFIERS:
      return 'rectifier'
    elif layer_label in MLP_SIGMOIDS:
      return 'sigmoid'
    else:
      return None


# Some common functions ================================================================
def _get_common_layer_labels():
  """ Returns layer labels common for all types of neural networks. """
  return ['ip', 'op']

def get_cnn_layer_labels(max_conv_size=7):
  """ Gets the layer labels for a CNN. """
  conv_layer_labels = ['fc', 'max-pool', 'avg-pool', 'softmax']
  for conv_size in range(3, max_conv_size+1, 2):
    conv_layer_labels.append('conv%d'%(conv_size))
    conv_layer_labels.append('res%d'%(conv_size))
  return _get_common_layer_labels() + conv_layer_labels

def get_mlp_layer_labels(class_or_reg, list_of_activations=None):
  """ Gets the layer labels for a Multi-layer perceptron. """
  list_of_activations = MLP_RECTIFIERS + MLP_SIGMOIDS \
    if list_of_activations is None else list_of_activations
  if class_or_reg == 'class' and 'softmax' not in list_of_activations:
    list_of_activations.append('softmax')
  elif class_or_reg == 'reg' and 'linear' not in list_of_activations:
    list_of_activations.append('linear')
  else:
    raise ValueError('class_or_reg should be either class or reg. Given %s.'%(
                     class_or_reg))
  return _get_common_layer_labels() + list_of_activations

