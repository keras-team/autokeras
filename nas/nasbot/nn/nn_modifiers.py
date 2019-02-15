"""
  Utilities to modify a given neural network and obtain a new one.
  --kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=star-args
# pylint: disable=too-many-branches

from argparse import Namespace
from copy import deepcopy
import numpy as np
# Local imports
from nas.nasbot.nn.neural_network import ConvNeuralNetwork, MultiLayerPerceptron, MLP_RECTIFIERS, \
                          MLP_SIGMOIDS, is_a_pooling_layer_label, is_a_conv_layer_label,\
                          CNNImageSizeMismatchException, CNNNoConvAfterIPException
from nas.nasbot.utils.general_utils import reorder_list_or_array, reorder_rows_and_cols_in_matrix
from nas.nasbot.utils.option_handler import get_option_specs, load_options
from nas.nasbot.utils.reporters import get_reporter

_DFLT_CHANGE_FRAC = 0.125
_DFLT_CHANGE_NUM_UNITS_SPAWN = 'all'
_DFLT_CHANGE_LAYERS_SPAWN = 20
_DFLT_NUM_SINGLE_STEP_MODIFICATIONS = 'all'
_DFLT_NUM_TWO_STEP_MODIFICATIONS = 0
_DFLT_NUM_THREE_STEP_MODIFICATIONS = 0
_DFLT_WEDGE_LAYER_CNN_CANDIDATES = ['conv3', 'conv5', 'conv7', 'res3', 'res5', 'res7']
_DFLT_WEDGE_LAYER_MLP_CANDIDATES = MLP_RECTIFIERS + MLP_SIGMOIDS
_DFLT_SIGMOID_SWAP = MLP_SIGMOIDS
_DFLT_RECTIFIER_SWAP = MLP_RECTIFIERS

_PRIMITIVE_PROB_MASSES = {'inc_single': 0.1,
                          'dec_single': 0.1,
                          'inc_en_masse': 0.1,
                          'dec_en_masse': 0.1,
                          'swap_layer': 0.2,
                          'wedge_layer': 0.1,
                          'remove_layer': 0.1,
                          'branch': 0.2,
                          'skip': 0.2,
                         }

nn_modifier_args = [
  # Change fractions for increasing the number of units in layers.
  get_option_specs('single_inc_change_frac', False, _DFLT_CHANGE_FRAC,
    'Default change fraction when increasing a single layer.'),
  get_option_specs('single_dec_change_frac', False, _DFLT_CHANGE_FRAC,
    'Default change fraction when decreasing a single layer.'),
  get_option_specs('en_masse_inc_change_frac', False, _DFLT_CHANGE_FRAC,
    'Default change fraction when increasing layers en_masse.'),
  get_option_specs('en_masse_dec_change_frac', False, _DFLT_CHANGE_FRAC,
    'Default change fraction when decreasing layers en_masse.'),
  # Number of networks to spawn by changing number of units in a single layer.
  get_option_specs('spawn_single_inc_num_units', False, _DFLT_CHANGE_NUM_UNITS_SPAWN,
    'Default number of networks to spawn by increasing # units in a single layer.'),
  get_option_specs('spawn_single_dec_num_units', False, _DFLT_CHANGE_NUM_UNITS_SPAWN,
    'Default number of networks to spawn by decreasing # units in a single layer.'),
  # Number of networks to spawn by adding or deleting a single layer.
  get_option_specs('spawn_add_layer', False, _DFLT_CHANGE_LAYERS_SPAWN,
    'Default number of networks to spawn by adding a layer.'),
  get_option_specs('spawn_del_layer', False, _DFLT_CHANGE_LAYERS_SPAWN,
    'Default number of networks to spawn by deleting a layer.'),
  # Number of double/triple step candidates - i.e. applications of basic primitives
  # twice/thrice before executing candidates
  get_option_specs('num_single_step_modifications', False,
    _DFLT_NUM_SINGLE_STEP_MODIFICATIONS,
    'Default number of networks to spawn via single step primitives.'),
  get_option_specs('num_two_step_modifications', False,
    _DFLT_NUM_TWO_STEP_MODIFICATIONS,
    'Default number of networks to spawn via two step primitives.'),
  get_option_specs('num_three_step_modifications', False,
    _DFLT_NUM_THREE_STEP_MODIFICATIONS,
    'Default number of networks to spawn via three step primitives.'),
  ]


# Generic utilities we will need in all functions below ==================================
def get_copies_from_old_nn(nn):
  """ Gets copies of critical parameters of the old network. """
  layer_labels = deepcopy(nn.layer_labels)
  num_units_in_each_layer = deepcopy(nn.num_units_in_each_layer)
  conn_mat = deepcopy(nn.conn_mat)
  mandatory_child_attributes = Namespace()
  for mca_str in nn.mandatory_child_attributes:
    mca_val = deepcopy(getattr(nn, mca_str))
    setattr(mandatory_child_attributes, mca_str, mca_val)
  return layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes

def get_new_nn(old_nn, layer_labels, num_units_in_each_layer, conn_mat,
               mandatory_child_attributes):
  """ Returns a new neural network of the same type as old_nn. """
  known_nn_class = True
  try:
    if old_nn.nn_class == 'cnn':
      new_cnn = ConvNeuralNetwork(layer_labels, conn_mat, num_units_in_each_layer,
                                  mandatory_child_attributes.strides,
                                  old_nn.all_layer_label_classes,
                                  old_nn.layer_label_similarities)
      return new_cnn
    elif old_nn.nn_class.startswith('mlp'):
      return MultiLayerPerceptron(old_nn.nn_class[4:], layer_labels, conn_mat,
                                  num_units_in_each_layer, old_nn.all_layer_label_classes,
                                  old_nn.layer_label_similarities)
    else:
      known_nn_class = False
  except (CNNImageSizeMismatchException, CNNNoConvAfterIPException, AssertionError):
    return None
  if not known_nn_class:
    raise ValueError('Unidentified nn_class %s.'%(old_nn.nn_class))

def add_layers_to_end_of_conn_mat(conn_mat, num_add_layers):
  """ Adds layers with no edges and returns. """
  new_num_layers = conn_mat.shape[0] + num_add_layers
  conn_mat.resize((new_num_layers, new_num_layers))
  return conn_mat


# Change architecture of the network
# ========================================================================================

# Add a layer ----------------------------------------------------------------------------
def wedge_layer(nn, layer_type, units_in_layer, layer_before, layer_after,
                new_layer_attributes=None):
  """ Wedges a layer of type layer_type after the layer given in layer_before. The
      output of the layer in layer_before goes to the new layer and the output of the
      new layer goes to layer_after. If an edge existed between layer_before and
      layer_after, it is removed. """
  layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes = \
    get_copies_from_old_nn(nn)
  layer_labels.append(layer_type)
  num_units_in_each_layer = np.append(num_units_in_each_layer, units_in_layer)
  if nn.nn_class == 'cnn':
    mandatory_child_attributes.strides.append(new_layer_attributes.stride)
  conn_mat = add_layers_to_end_of_conn_mat(conn_mat, 1)
  conn_mat[layer_before, -1] = 1
  conn_mat[-1, layer_after] = 1
  conn_mat[layer_before, layer_after] = 0
  return get_new_nn(nn, layer_labels, num_units_in_each_layer, conn_mat,
                    mandatory_child_attributes)

def _get_non_None_elements(iter_of_vals):
  """ Returns non None values. """
  return [x for x in iter_of_vals if x is not None]

def _determine_num_units_for_wedge_layer(nn, edge):
  """ Determines the number of layers for wedging a layer. This is usually the average
      of the parent (edge[0]) and child (edge[1]).
  """
  edge_num_layers = _get_non_None_elements(
                      [nn.num_units_in_each_layer[idx] for idx in edge])
  if len(edge_num_layers) > 0:
    return round(sum(edge_num_layers) / len(edge_num_layers))
  else:
    parents = nn.get_parents(edge[0])
    if len(parents) == 0:
      # Means you have reached the input node
      ip_children = nn.get_children(edge[0])
      children_num_units = _get_non_None_elements(
                             [nn.num_units_in_each_layer[ch] for ch in ip_children])
      if len(children_num_units) == 0:
        # Create a layer with 16 units
        children_num_units = [16]
      return sum(children_num_units) / len(children_num_units)
    else:
      parent_num_units = _get_non_None_elements(
                          [nn.num_units_in_each_layer[par] for par in parents])
      if len(parent_num_units) > 0:
        return sum(parent_num_units) / len(parent_num_units)
      else:
        par_num_units = []
        for par in parents:
          par_num_units.append(_determine_num_units_for_wedge_layer(nn, (par, edge[0])))
        par_num_units = _get_non_None_elements(par_num_units)
        return sum(par_num_units) / len(par_num_units)

def get_list_of_wedge_layer_modifiers(nn, num_modifications='all',
                                      internal_layer_type_candidates=None,
                                      choose_pool_with_prob=0.05,
                                      choose_stride_2_with_prob=0.05):
  """ Returns a list of operations for adding a layer in between two layers. """
  # A local function for creating a modifier
  def _get_wedge_modifier(_layer_type, _num_units, _edge, _nl_attributes):
    """ Returns a modifier which wedges an edge between the edge. """
    return lambda arg_nn: wedge_layer(arg_nn, _layer_type, _num_units,
                                       _edge[0], _edge[1], _nl_attributes)
  # Pre-process arguments
  nn_is_a_cnn = nn.nn_class == 'cnn'
  if internal_layer_type_candidates is None:
    if nn_is_a_cnn:
      internal_layer_type_candidates = _DFLT_WEDGE_LAYER_CNN_CANDIDATES
    else:
      internal_layer_type_candidates = _DFLT_WEDGE_LAYER_MLP_CANDIDATES
  if not nn_is_a_cnn:
    choose_pool_with_prob = 0
  all_edges = nn.get_edges()
  num_modifications = len(all_edges) if num_modifications == 'all' else num_modifications
  op_layer_idx = nn.get_op_layer_idx() # Output layer
  ip_layer_idx = nn.get_ip_layer_idx() # Input layer
  # We won't change this below so keep it as it is
  nonconv_nl_attrs = Namespace(stride=None)
  conv_nl_attrs_w_stride_1 = Namespace(stride=1)
  conv_nl_attrs_w_stride_2 = Namespace(stride=2)
  # Iterate through all edges
  ret = []
  for edge in all_edges:
    curr_layer_type = None
    # First handle the edges cases
    if edge[1] == op_layer_idx:
      continue
    elif nn_is_a_cnn and nn.layer_labels[edge[0]] == 'fc':
      curr_layer_type = 'fc'
      curr_num_units = nn.num_units_in_each_layer[edge[0]]
      nl_attrs = nonconv_nl_attrs
    elif not nn_is_a_cnn and edge[1] == op_layer_idx:
      # Don't add new layers just before the output for MLPs
      continue
    elif edge[0] == ip_layer_idx and nn_is_a_cnn:
      curr_pool_prob = 0 # No pooling layer right after the input for a CNN
    else:
      curr_pool_prob = choose_pool_with_prob

    if curr_layer_type is None:
      if np.random.random() < curr_pool_prob:
        curr_layer_candidates = ['avg-pool', 'max-pool']
      else:
        curr_layer_candidates = internal_layer_type_candidates
      curr_layer_type = np.random.choice(curr_layer_candidates, 1)[0]
      if curr_layer_type in ['max-pool', 'avg-pool', 'linear', 'softmax']:
        curr_num_units = None
      else:
        curr_num_units = _determine_num_units_for_wedge_layer(nn, edge)
      # Determine stride
      if is_a_conv_layer_label(curr_layer_type):
        nl_attrs = conv_nl_attrs_w_stride_2 if \
          np.random.random() < choose_stride_2_with_prob else conv_nl_attrs_w_stride_1
      else:
        nl_attrs = nonconv_nl_attrs
    ret.append(_get_wedge_modifier(curr_layer_type, curr_num_units, edge, nl_attrs))
    # Break if more than the number of modifications
    if len(ret) >= num_modifications:
      break
  return ret


# Removing a layer -----------------------------------------------------------------------
def remove_layer(nn, del_idx, additional_edges, new_strides=None):
  """ Deletes the layer indicated in del_idx and adds additional_edges specified
      in additional_edges. """
  layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes = \
    get_copies_from_old_nn(nn)
  # First add new edges to conn_mat and remove edges to and from del_idx
  for add_edge in additional_edges:
    conn_mat[add_edge[0], add_edge[1]] = 1
  conn_mat[del_idx, :] = 0
  conn_mat[:, del_idx] = 0
  # Now reorder everything so that del_idx is at the end
  all_idxs = list(range(len(layer_labels)))
  new_order = all_idxs[:del_idx] + all_idxs[del_idx+1:] + [del_idx]
  # Now reorder everything so that the layer to be remove is at the end
  layer_labels = reorder_list_or_array(layer_labels, new_order)
  num_units_in_each_layer = reorder_list_or_array(num_units_in_each_layer, new_order)
  conn_mat = reorder_rows_and_cols_in_matrix(conn_mat, new_order)
  # remove layer
  layer_labels = layer_labels[:-1]
  num_units_in_each_layer = num_units_in_each_layer[:-1]
  conn_mat = conn_mat[:-1, :-1]
  # Strides for a convolutional network
  if nn.nn_class == 'cnn':
    new_strides = new_strides if new_strides is not None else \
                  mandatory_child_attributes.strides
    mandatory_child_attributes.strides = reorder_list_or_array(
      new_strides, new_order)
    mandatory_child_attributes.strides = mandatory_child_attributes.strides[:-1]
  return get_new_nn(nn, layer_labels, num_units_in_each_layer, conn_mat,
                    mandatory_child_attributes)


def get_list_of_remove_layer_modifiers(old_nn):
  """ Returns a list of primitives which remove a layer from a neural network. """
  # pylint: disable=too-many-locals
  # A local function to return the modifier
  if old_nn.num_processing_layers == 0:
    # Don't delete any layers if there are no processing layers.
    return []
  def _get_remove_modifier(_del_idx, _add_edges, *_args, **_kwargs):
    """ Returns a modifier which deletes _del_idx and adds _add_edges. """
    return lambda arg_nn: remove_layer(arg_nn, _del_idx, _add_edges, *_args, **_kwargs)
  # Now check every layer
  ret = []
  for idx, ll in enumerate(old_nn.layer_labels):
    if ll in ['ip', 'op', 'softmax', 'linear']: # Don't delete any of these layers
      continue
    curr_parents = old_nn.get_parents(idx)
    parent_labels = [old_nn.layer_labels[par_idx] for par_idx in curr_parents]
    if ll == 'fc' and (not parent_labels == ['fc'] * len(parent_labels)):
      # If the parents of a fc layer are also not fc then do not delete
      continue
    curr_children = old_nn.get_children(idx)
    if old_nn.nn_class == 'cnn' and \
       old_nn.pre_img_inv_sizes[idx] != old_nn.post_img_inv_sizes[idx]:
      change_stride_idxs = None
      # First check if the children are modifiable
      child_strides = [old_nn.strides[ch_idx] for ch_idx in curr_children]
      if child_strides == [1] * len(curr_children):
        change_stride_idxs = curr_children # we will change the strides of the children
      if change_stride_idxs is None:
        parent_strides = [old_nn.strides[par_idx] for par_idx in curr_parents]
        if parent_strides == [1] * len(curr_parents):
          change_stride_idxs = curr_parents
      # If we have successfuly identified children/parents which we can modify, great!
      # Otherwise, lets not change anything and hope that it
      # does not break anything. If it does, there is an exception to handle this.
      if change_stride_idxs is not None:
        new_strides = deepcopy(old_nn.strides)
        for csi in change_stride_idxs:
          new_strides[csi] = 2
      else:
        new_strides = None
    else:
      new_strides = None
    # Now delete the layer and add new adges
    num_children_on_each_parent = [len(old_nn.get_children(par_idx)) for par_idx in
                                   curr_parents]
    num_parents_on_each_child = [len(old_nn.get_parents(ch_idx)) for ch_idx in
                                 curr_children]
    must_add_children = [curr_children[i] for i in range(len(curr_children)) if
                         num_parents_on_each_child[i] == 1]
    must_add_parents = [curr_parents[i] for i in range(len(curr_parents)) if
                        num_children_on_each_parent[i] == 1]
    num_must_add_children = len(must_add_children)
    num_must_add_parents = len(must_add_parents)
    np.random.shuffle(must_add_children)
    np.random.shuffle(must_add_parents)
    add_edges = []
    for _ in range(min(num_must_add_children, num_must_add_parents)):
      add_edges.append((must_add_parents.pop(), must_add_children.pop()))
    # Add edges for left over children/parents
    if num_must_add_children > num_must_add_parents:
      diff = num_must_add_children - num_must_add_parents
      cand_parents = list(np.random.choice(curr_parents, diff))
      for _ in range(diff):
        add_edges.append((cand_parents.pop(), must_add_children.pop()))
    if num_must_add_parents > num_must_add_children:
      diff = num_must_add_parents - num_must_add_children
      cand_children = list(np.random.choice(curr_children, diff))
      for _ in range(diff):
        add_edges.append((must_add_parents.pop(), cand_children.pop()))
    ret.append(_get_remove_modifier(idx, add_edges, new_strides))
  return ret


# Branching modifications ----------------------------------------------------------------
def create_duplicate_branch(nn, path, keep_layer_with_prob=0.5):
  """ Creates a new network which creates a new branch between path[0] and path[-1] and
      copies all layers between. It keeps a layer in between with probability 0.5. If
      in CNNs, the layer shrinks the size of the image, then we keep it with prob 1.
  """
  layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes = \
    get_copies_from_old_nn(nn)
  # First decide which nodes in the path to keep
  branched_path = [path[0]]
  fc_encountered = False
  for idx in path[1: -1]:
    if idx == path[1] and nn.get_ip_layer_idx() == path[0]:
      branched_path.append(idx) # Append if the branch starts at ip and this is a child.
    elif idx == path[-2] and len(branched_path) == 1:
      branched_path.append(idx) # If this is the last layer and we have not appended yet.
    elif is_a_pooling_layer_label(nn.layer_labels[idx]) or \
      nn.layer_labels[idx] in ['linear', 'softmax'] or \
      (hasattr(nn, 'strides') and nn.strides[idx] == 2):
      branched_path.append(idx)
    elif nn.layer_labels[idx] == 'fc' and not fc_encountered:
      branched_path.append(idx)
      fc_encountered = True
    elif np.random.random() < keep_layer_with_prob:
      branched_path.append(idx)
  branched_path.append(path[-1])
  # Now create additional nodes
  num_new_nodes = len(branched_path) - 2
  layer_labels.extend([nn.layer_labels[idx] for idx in branched_path[1:-1]])
  num_units_in_each_layer = np.concatenate((num_units_in_each_layer,
    [nn.num_units_in_each_layer[idx] for idx in branched_path[1:-1]]))
  # Add edges
  new_idxs = list(range(nn.num_layers, nn.num_layers + num_new_nodes))
  conn_mat = add_layers_to_end_of_conn_mat(conn_mat, num_new_nodes)
  if num_new_nodes == 0:
    conn_mat[branched_path[0], branched_path[1]] = 1
  else:
    conn_mat[branched_path[0], new_idxs[0]] = 1
    conn_mat[new_idxs[-1], branched_path[-1]] = 1
    for new_idx in new_idxs[:-1]:
      conn_mat[new_idx, new_idx + 1] = 1
  # Add strides
  if nn.nn_class == 'cnn':
    mandatory_child_attributes.strides.extend([nn.strides[idx] for idx in
                                               branched_path[1:-1]])
  return get_new_nn(nn, layer_labels, num_units_in_each_layer, conn_mat,
                    mandatory_child_attributes)


def _get_path_for_branching_from_start_layer(nn, start_layer, min_path_length=4,
                                             end_path_prob=0.20):
  """ Returns a path which starts at start layer. """
  path = [start_layer]
  while True:
    curr_layer = path[-1]
    curr_children = nn.get_children(curr_layer)
    next_layer = int(np.random.choice(curr_children, 1))
    path.append(next_layer)
    if nn.layer_labels[next_layer] == 'op':
      break
    elif len(path) < min_path_length:
      pass
    elif np.random.random() < end_path_prob:
      break
  return path


def _get_start_layer_probs_for_branching_and_skipping(nn):
  """ Returns probabilities for the start layer to be used in branching and
      skipping primitives.
  """
  dists_from_ip = nn.get_distances_from_ip()
  start_layer_prob = []
  for layer_idx, layer_label in enumerate(nn.layer_labels):
    # We pick the first layer with distance inversely proportional to its distance from ip
    curr_layer_prob = 0 if layer_label in ['op', 'softmax', 'fc', 'linear'] else \
                      1.0 / np.sqrt(1 + dists_from_ip[layer_idx])
    start_layer_prob.append(curr_layer_prob)
  start_layer_prob = np.array(start_layer_prob)
  return start_layer_prob


def get_list_of_branching_modifiers(nn, num_modifiers=None, **kwargs):
  """ Returns a list of operators for Neural networks that create branches in the
      architecture.
  """
  if nn.num_processing_layers == 0:
    # Don't create any branches if there are no processing layers.
    return []
  # Define a local function to return the modifier
  def _get_branching_modifier(_path, *_args, **_kwargs):
    """ Returns a modifier which duplicates the path along _path. """
    return lambda arg_nn: create_duplicate_branch(arg_nn, _path, *_args, **_kwargs)
  # Some preprocessing
  num_modifiers = num_modifiers if num_modifiers is not None else 2 * nn.num_layers
  start_layer_prob = _get_start_layer_probs_for_branching_and_skipping(nn)
  ret = []
  if sum(start_layer_prob) <= 0.0:
    return ret # return immediately with an empty list
  while len(ret) < num_modifiers:
    start_layer_prob = start_layer_prob / sum(start_layer_prob)
    start_layer = int(np.random.choice(nn.num_layers, 1, p=start_layer_prob))
    path = _get_path_for_branching_from_start_layer(nn, start_layer)
    start_layer_prob[start_layer] *= 0.9 # shrink probability of picking this layer again.
    ret.append(_get_branching_modifier(path, **kwargs))
  return ret


# Skipping modifications -----------------------------------------------------------------
def create_skipped_network(nn, start_layer, end_layer, pool_layer_type='avg'):
  """ Creates a new layer with a skip connection from start_layer to end_layer.
      In a CNN, if the image sizes do not match, this creates additional pooling layers
      (either avg-pool or max-pool) to make them match.
  """
  layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes = \
    get_copies_from_old_nn(nn)
  if nn.nn_class != 'cnn' or \
    nn.post_img_inv_sizes[start_layer] == nn.pre_img_inv_sizes[end_layer]:
    conn_mat[start_layer, end_layer] = 1
  else:
    # Determine number of new layers, the number of units and strides in each layer.
    num_new_pool_layers = int(np.log2(nn.pre_img_inv_sizes[end_layer] /
                                      nn.post_img_inv_sizes[start_layer]))
    new_layer_idxs = list(range(nn.num_layers, nn.num_layers + num_new_pool_layers))
    num_units_in_each_layer = np.concatenate((num_units_in_each_layer,
      [None] * num_new_pool_layers))
    mandatory_child_attributes.strides.extend([None] * num_new_pool_layers)
    # Determine layer labels
    if pool_layer_type.lower().startswith('avg'):
      new_layer_type = 'avg-pool'
    elif pool_layer_type.lower().startswith('max'):
      new_layer_type = 'max-pool'
    else:
      raise ValueError('pool_layer_type should be \'avg\' or \'max\'.')
    new_layer_labels = [new_layer_type for _ in range(num_new_pool_layers)]
    layer_labels.extend(new_layer_labels)
    conn_mat = add_layers_to_end_of_conn_mat(conn_mat, num_new_pool_layers)
    # Finally, the conn_mat
    conn_mat[start_layer, new_layer_idxs[0]] = 1
    conn_mat[new_layer_idxs[-1], end_layer] = 1
    for new_idx in new_layer_idxs[:-1]:
      conn_mat[new_idx, new_idx + 1] = 1
  return get_new_nn(nn, layer_labels, num_units_in_each_layer, conn_mat,
                    mandatory_child_attributes)


def _get_end_layer_probs_for_skipping(nn, start_layer):
  """ Returns the end layer probabilities to be used in skipping. """
  dists_from_ip = nn.get_distances_from_ip()
  dists_to_op = nn.get_distances_to_op()
  is_a_cnn = nn.nn_class.startswith('cnn')
  end_layer_prob = []
  for layer_idx, layer_label in enumerate(nn.layer_labels):
    curr_layer_prob = 'assign'
    if dists_from_ip[layer_idx] - 1 <= dists_from_ip[start_layer] or \
       dists_to_op[layer_idx] + 1 >= dists_to_op[start_layer] or \
       layer_label in ['ip', 'op', 'softmax']:
      curr_layer_prob = 'no-assign'
    elif is_a_cnn and \
         nn.post_img_inv_sizes[start_layer] > nn.pre_img_inv_sizes[layer_idx]:
      # If the layer has an input image size *larger* than the output of the
      # start layer, then do not assign.
      curr_layer_prob = 'no-assign'
    elif layer_label == 'fc':
      # If its a fully connected layer, connect with this only if it is the first
      # fc layer.
      curr_layer_parent_labels = [nn.layer_labels[x] for x in nn.get_parents(layer_idx)]
      if not all([(is_a_pooling_layer_label(clpl) or is_a_conv_layer_label(clpl)) for
                  clpl in curr_layer_parent_labels]):
        curr_layer_prob = 'no-assign'
    curr_layer_prob = 0.0 if curr_layer_prob == 'no-assign' else 1.0
    end_layer_prob.append(curr_layer_prob)
  if sum(end_layer_prob) == 0:
    return None
  else:
    end_layer_prob = np.array(end_layer_prob)
    end_layer_prob = end_layer_prob / end_layer_prob.sum()
    return end_layer_prob
  return


def get_list_of_skipping_modifiers(nn, num_modifiers=None, **kwargs):
  """ Returns a list of operators for Neural networks that create branches in the
      architecture.
  """
  # Define a local function to return the modifier
  def _get_skipping_modifier(_start_layer, _end_layer, **_kwargs):
    """ Returns a modifier which adds a skip connected from start_layer to end_layer. """
    return lambda arg_nn: create_skipped_network(arg_nn, _start_layer, _end_layer,
                                                 **_kwargs)
  # Some preprocessing
  num_modifiers = num_modifiers if num_modifiers is not None else nn.num_layers
  max_num_tries = 2 * num_modifiers
  start_layer_prob = _get_start_layer_probs_for_branching_and_skipping(nn)
  start_layer_prob[0] = 0.0
  ret = []
  if sum(start_layer_prob) <= 0.0:
    return ret # return immediately with an empty list
  for _ in range(max_num_tries):
    start_layer_prob = start_layer_prob / sum(start_layer_prob)
    start_layer = int(np.random.choice(nn.num_layers, 1, p=start_layer_prob))
    end_layer_prob = _get_end_layer_probs_for_skipping(nn, start_layer)
    if end_layer_prob is None:
      continue
    end_layer = int(np.random.choice(nn.num_layers, 1, p=end_layer_prob))
    ret.append(_get_skipping_modifier(start_layer, end_layer, **kwargs))
    if len(ret) >= num_modifiers:
      break
  return ret


# Swapping -------------------------------------------------------------------------------
def swap_layer_type(nn, layer_idx, replace_with, new_stride):
  """ Swaps out the type of layer in layer_idx with replace_with """
  if nn.layer_labels[layer_idx] == replace_with:
    raise ValueError('Cannot replace layer %d with the same layer type (%s).'%(
                     layer_idx, replace_with))
  layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes = \
    get_copies_from_old_nn(nn)
  layer_labels[layer_idx] = replace_with
  if nn.nn_class == 'cnn':
    if is_a_pooling_layer_label(replace_with):
      num_units_in_each_layer[layer_idx] = None
    mandatory_child_attributes.strides[layer_idx] = new_stride
  return get_new_nn(nn, layer_labels, num_units_in_each_layer, conn_mat,
                    mandatory_child_attributes)

def get_list_of_swap_layer_modifiers(nn, num_modifications='all',
                                     change_stride_with_prob=0.05,
                                     rectifier_swap_candidates=None,
                                     sigmoid_swap_candidates=None):
  """ Returns a list of modifiers for swapping a layer with another. """
  # pylint: disable=too-many-statements
  # Define a local function to return the modifier
  def _get_swap_layer_modifier(_layer_idx, _replace_with, _new_stride):
    """ Returns a modifier for swapping a layer. """
    return lambda arg_nn: swap_layer_type(arg_nn, _layer_idx, _replace_with, _new_stride)
  # Preprocessing
  if nn.nn_class.startswith('mlp'):
    rectifier_swap_candidates = rectifier_swap_candidates if \
      rectifier_swap_candidates is not None else _DFLT_RECTIFIER_SWAP
    sigmoid_swap_candidates = sigmoid_swap_candidates if \
      sigmoid_swap_candidates is not None else _DFLT_SIGMOID_SWAP
  # Determine the order of the layers
  layer_order = list(range(nn.num_layers))
  if num_modifications == 'all' or num_modifications >= nn.num_layers:
    num_modifications = nn.num_layers
  else:
    np.random.shuffle(layer_order)
  # iterate through the layers and return
  ret = []
  for idx in layer_order:
    ll = nn.layer_labels[idx]
    if ll in ['ip', 'op', 'fc', 'softmax', 'linear']:
      continue # don't swap out any of these
    # Determine candidates for swapping out
    if ll == 'conv3':
      candidates = ['res3', 'res5', 'conv5', 'conv7', 'max-pool', 'avg-pool']
      cand_probs = [0.25, 0.25, 0.15, 0.25, 0.05, 0.05]
    elif ll == 'conv5':
      candidates = ['res3', 'res5', 'conv3', 'conv7', 'max-pool', 'avg-pool']
      cand_probs = [0.25, 0.25, 0.2, 0.2, 0.05, 0.05]
    elif ll == 'conv7':
      candidates = ['res3', 'res5', 'conv3', 'conv5', 'max-pool', 'avg-pool']
      cand_probs = [0.25, 0.25, 0.25, 0.15, 0.05, 0.05]
    elif ll == 'conv9':
      candidates = ['res3', 'res5', 'conv3', 'conv5', 'conv7', 'max-pool', 'avg-pool']
      cand_probs = [0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05]
    elif ll == 'res3':
      candidates = ['conv3', 'conv5', 'res5', 'res7', 'max-pool', 'avg-pool']
      cand_probs = [0.25, 0.25, 0.15, 0.25, 0.05, 0.05]
    elif ll == 'res5':
      candidates = ['conv3', 'conv5', 'res3', 'res7', 'max-pool', 'avg-pool']
      cand_probs = [0.25, 0.25, 0.2, 0.2, 0.05, 0.05]
    elif ll == 'res7':
      candidates = ['conv3', 'conv5', 'res3', 'res5', 'max-pool', 'avg-pool']
      cand_probs = [0.25, 0.25, 0.25, 0.15, 0.05, 0.05]
    elif ll == 'res9':
      candidates = ['conv3', 'conv5', 'res3', 'res5', 'res7', 'max-pool', 'avg-pool']
      cand_probs = [0.2, 0.2, 0.2, 0.2, 0.1, 0.05, 0.05]
    elif ll == 'avg-pool':
      candidates = ['max-pool']
      cand_probs = None
    elif ll == 'max-pool':
      candidates = ['avg-pool']
      cand_probs = None
    elif ll in MLP_RECTIFIERS:
      candidates = sigmoid_swap_candidates
      cand_probs = None
    elif ll in MLP_SIGMOIDS:
      candidates = rectifier_swap_candidates
      cand_probs = None
    else:
      raise ValueError('Unidentified layer_type: %s.'%(ll))
    # I am determining the probabilities above completely ad-hoc for reasons I don't
    # know why.
    # Choose replace_with
    if cand_probs is not None:
      cand_probs = np.array(cand_probs)
      cand_probs = cand_probs / cand_probs.sum()
    replace_with = np.random.choice(candidates, 1, p=cand_probs)[0]
    # Determine the stride
    if nn.nn_class == 'cnn':
      if is_a_pooling_layer_label(replace_with):
        new_stride = None
      elif is_a_conv_layer_label(replace_with) and is_a_pooling_layer_label(ll):
        new_stride = 2
      elif is_a_conv_layer_label(ll) and np.random.random() < change_stride_with_prob:
        new_stride = 1 if nn.strides[idx] == 2 else 2
      else:
        new_stride = nn.strides[idx]
    else:
      new_stride = None
    # Create modifier and append
    ret.append(_get_swap_layer_modifier(idx, replace_with, new_stride))
    if len(ret) >= num_modifications:
      break # Check if you have exceeded the maximum amount
  return ret

# Change number of units in a layer
# ========================================================================================
def change_num_units_in_layers(nn, change_layer_idxs, change_layer_vals):
  """ Changes the number of units in change_layer_idxs to change_layer_vals. """
  layer_labels, num_units_in_each_layer, conn_mat, mandatory_child_attributes = \
    get_copies_from_old_nn(nn)
  for i, ch_idx in enumerate(change_layer_idxs):
    if is_a_pooling_layer_label(layer_labels[ch_idx]):
      raise ValueError('Asked to change a pooling layer value. This is not allowed.')
    else:
      num_units_in_each_layer[ch_idx] = change_layer_vals[i]
  return get_new_nn(nn, layer_labels, num_units_in_each_layer, conn_mat,
                    mandatory_child_attributes)

def _get_directly_modifable_layer_idxs(nn):
  """ Returns indices that are directly modifiable in the network. """
  if isinstance(nn, ConvNeuralNetwork):
    return [i for i in range(nn.num_layers) if (nn.layer_labels[i].startswith('res') or
                                                nn.layer_labels[i].startswith('conv') or
                                                nn.layer_labels[i] == 'fc')]
  elif isinstance(nn, MultiLayerPerceptron):
    return [i for i in range(nn.num_layers) if nn.layer_labels[i] in
                                               MLP_RECTIFIERS + MLP_SIGMOIDS]
  else:
    raise ValueError('Unidentified nn type: %s'%(nn.nn_class))

def _get_change_ratio_from_change_frac(change_frac, inc_or_dec):
  """ Gets the change ratio from the change fraction, i.e. 1 +/- change_frac depending
      on inc_or_dec. """
  if inc_or_dec.lower() == 'increase':
    return 1 + abs(change_frac)
  elif inc_or_dec.lower() == 'decrease':
    return 1 - abs(change_frac)
  else:
    raise ValueError('change_ratio should be one of \'increase\' or \'decrease\'.')

def modify_several_layers(nn, inc_or_dec, layer_group_desc,
                          change_frac=_DFLT_CHANGE_FRAC):
  """ A function to increase or decrease several layers at the same time. inc_or_dec
      is a string with values 'increase' or 'decrease' and layer_group_desc is a string
      with one of the following values: '1/2', '2/2', '1/4', ... '4/4', '1/8', ... 8/8."""
  change_ratio = _get_change_ratio_from_change_frac(change_frac, inc_or_dec)
  modifiable_layers = _get_directly_modifable_layer_idxs(nn)
  num_modifiable_layers = len(modifiable_layers)
  # Now decide which groups to change
  num_groups = int(layer_group_desc[-1])
  group_idx = int(layer_group_desc[0])
  start_idx = (group_idx - 1) * num_modifiable_layers / num_groups
  end_idx = group_idx * num_modifiable_layers / num_groups
  modify_layer_idxs = modifiable_layers[start_idx:end_idx]
  modify_vals = [round(change_ratio * nn.num_units_in_each_layer[i])
                 for i in modify_layer_idxs]
  return change_num_units_in_layers(nn, modify_layer_idxs, modify_vals)

def modify_num_units_on_random_nodes(nn, num_units, inc_or_dec,
                                     change_frac=_DFLT_CHANGE_FRAC):
  """ A function to increase or decrase the number of units in a single layer. """
  change_ratio = _get_change_ratio_from_change_frac(change_frac, inc_or_dec)
  modifiable_layers = _get_directly_modifable_layer_idxs(nn)
  modify_layer_idxs = np.random.choice(modifiable_layers,
                                       max(num_units, modifiable_layers))
  modify_vals = [round(change_ratio * nn.num_units_in_each_layer[i])
                 for i in modify_layer_idxs]
  return change_num_units_in_layers(nn, modify_layer_idxs, modify_vals)

def _get_candidate_layers_for_modifying_num_units(nn, num_candidates='all'):
  """ Returns a set of candidate layers for modifying the number of units. """
  modifiable_layers = _get_directly_modifable_layer_idxs(nn)
  if num_candidates == 'all' or num_candidates >= len(modifiable_layers):
    return modifiable_layers
  else:
    np.random.shuffle(modifiable_layers)
    return modifiable_layers[:num_candidates]

def get_list_of_single_layer_modifiers(old_nn, inc_or_dec, num_layers_to_modify='all',
                                       change_frac=_DFLT_CHANGE_FRAC):
  """ Returns a list of primitives which change old_nn in one layer. """
  # Define a local function to obtain the modifier
  def _get_modifier(_ltm, _change_val):
    """ Gets a modifier with the current values of ltm and change_val. """
    return lambda nn: change_num_units_in_layers(nn, [_ltm], [_change_val])
  # The problem with doing ret.append(lamda nn ....(nn, ltm, change_val)) is that python
  # uses the current values of the variables ltm and change_val when calling - and this
  # turns out to be the last value in layers_to_modify. i.e. Python looks up the
  # variable name at the time the function is called, not when it is created.
  # See this:
  # https://stackoverflow.com/questions/10452770/python-lambdas-binding-to-local-values
  change_ratio = _get_change_ratio_from_change_frac(change_frac, inc_or_dec)
  layers_to_modify = _get_candidate_layers_for_modifying_num_units(old_nn,
                       num_layers_to_modify)
  ret = []
  for ltm in layers_to_modify:
    change_val = round(change_ratio * old_nn.num_units_in_each_layer[ltm])
    ret.append(_get_modifier(ltm, change_val))
  return ret

# Define the following for convenience
# Increase num units en masse
increase_en_masse_1_2 = lambda nn, *a: modify_several_layers(nn, 'increase', '1/2', *a)
increase_en_masse_2_2 = lambda nn, *a: modify_several_layers(nn, 'increase', '2/2', *a)
increase_en_masse_1_4 = lambda nn, *a: modify_several_layers(nn, 'increase', '1/4', *a)
increase_en_masse_2_4 = lambda nn, *a: modify_several_layers(nn, 'increase', '2/4', *a)
increase_en_masse_3_4 = lambda nn, *a: modify_several_layers(nn, 'increase', '3/4', *a)
increase_en_masse_4_4 = lambda nn, *a: modify_several_layers(nn, 'increase', '4/4', *a)
increase_en_masse_1_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '1/8', *a)
increase_en_masse_2_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '2/8', *a)
increase_en_masse_3_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '3/8', *a)
increase_en_masse_4_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '4/8', *a)
increase_en_masse_5_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '5/8', *a)
increase_en_masse_6_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '6/8', *a)
increase_en_masse_7_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '7/8', *a)
increase_en_masse_8_8 = lambda nn, *a: modify_several_layers(nn, 'increase', '8/8', *a)
# Decrease num units en masse
decrease_en_masse_1_2 = lambda nn, *a: modify_several_layers(nn, 'decrease', '1/2', *a)
decrease_en_masse_2_2 = lambda nn, *a: modify_several_layers(nn, 'decrease', '2/2', *a)
decrease_en_masse_1_4 = lambda nn, *a: modify_several_layers(nn, 'decrease', '1/4', *a)
decrease_en_masse_2_4 = lambda nn, *a: modify_several_layers(nn, 'decrease', '2/4', *a)
decrease_en_masse_3_4 = lambda nn, *a: modify_several_layers(nn, 'decrease', '3/4', *a)
decrease_en_masse_4_4 = lambda nn, *a: modify_several_layers(nn, 'decrease', '4/4', *a)
decrease_en_masse_1_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '1/8', *a)
decrease_en_masse_2_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '2/8', *a)
decrease_en_masse_3_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '3/8', *a)
decrease_en_masse_4_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '4/8', *a)
decrease_en_masse_5_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '5/8', *a)
decrease_en_masse_6_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '6/8', *a)
decrease_en_masse_7_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '7/8', *a)
decrease_en_masse_8_8 = lambda nn, *a: modify_several_layers(nn, 'decrease', '8/8', *a)


def get_list_of_en_masse_change_primitives(nn, inc_or_dec='incdec'):
  """ Returns the list of primitives which changes the number of units en masse. """
  ret = []
  # Change 1/2
  if nn.num_internal_layers >= 4:
    if 'inc' in inc_or_dec:
      ret.extend([increase_en_masse_1_2, # increase
                  increase_en_masse_2_2,
                 ])
    if 'dec' in inc_or_dec:
      ret.extend([decrease_en_masse_1_2, # decrease
                  decrease_en_masse_2_2,
                 ])
  # Change 1/4
  if nn.num_internal_layers >= 8:
    if 'inc' in inc_or_dec:
      ret.extend([increase_en_masse_1_4, # increase
                  increase_en_masse_2_4,
                  increase_en_masse_3_4,
                  increase_en_masse_4_4,
                 ])
    if 'dec' in inc_or_dec:
      ret.extend([decrease_en_masse_1_4, # decrease
                  decrease_en_masse_2_4,
                  decrease_en_masse_3_4,
                  decrease_en_masse_4_4,
                 ])
  # Change 1/8
  if nn.num_internal_layers >= 16:
    if 'inc' in inc_or_dec:
      ret.extend([increase_en_masse_1_8, # increase
                  increase_en_masse_2_8,
                  increase_en_masse_3_8,
                  increase_en_masse_4_8,
                  increase_en_masse_5_8,
                  increase_en_masse_6_8,
                  increase_en_masse_7_8,
                  increase_en_masse_8_8,
                  ])
    if 'dec' in inc_or_dec:
      ret.extend([decrease_en_masse_1_8, # decrease
                  decrease_en_masse_2_8,
                  decrease_en_masse_3_8,
                  decrease_en_masse_4_8,
                  decrease_en_masse_5_8,
                  decrease_en_masse_6_8,
                  decrease_en_masse_7_8,
                  decrease_en_masse_8_8,
                  ])
  return ret

# A class to put it all together
# ========================================================================================
class NNModifier(object):
  """ A class for modifying a neural nework using many of the operations above. """

  def __init__(self, constraint_checker=None, options=None, reporter=None):
    """ Constructor. """
    self.reporter = get_reporter(reporter)
    self.constraint_checker = constraint_checker
    if options is None:
      options = load_options(nn_modifier_args)
    self.options = options

  def __call__(self, list_of_nns, num_modifications, num_steps_probs, max_num_steps=None,
               **kwargs):
    """ Takes a list of neural network nns and applies the library of changes to it to
        produce a list of modifications.
        num_steps_probs is a list of probabilities that indicate with what probability
        we want to use a certain number of steps.
    """
    # Preprocessing
    if not hasattr(list_of_nns, '__iter__'):
      list_of_nns = [list_of_nns]
    # determine num_steps_probs
    if num_steps_probs is None and isinstance(max_num_steps, (int, long, float)):
      num_steps_probs = np.ones((max_num_steps,))/float(max_num_steps)
    elif isinstance(num_steps_probs, (int, long, float)):
      num_steps_probs = np.zeros((num_steps_probs,))
      num_steps_probs[-1] = 1.0
    # Determine how many modifications per nn
    nn_idxs = list(range(len(list_of_nns)))
    if hasattr(num_modifications, '__iter__'):
      num_modifs_for_each_nn = num_modifications
    else:
      nn_choices_for_modif = np.random.choice(nn_idxs, num_modifications, replace=True)
      num_modifs_for_each_nn = [np.sum(nn_choices_for_modif == i) for i in nn_idxs]
    # Create a list
    ret = []
    for idx in nn_idxs:
      ret.extend(self.get_modifications_for_a_single_nn(list_of_nns[idx],
        num_modifs_for_each_nn[idx], num_steps_probs, **kwargs))
    return ret

  def get_modifications_for_a_single_nn(self, nn, num_modifications, num_steps_probs,
                                        **kwargs):
    """ Takes a neural network nn and applies the library of changes to it to produce
        a list of modifiers.
        num_steps_probs is a list of probabilities that indicate with what probability
        we want to use a certain number of steps.
    """
    max_num_steps = len(num_steps_probs)
    num_step_choices = np.random.choice(list(range(max_num_steps)), num_modifications,
                                        replace=True, p=num_steps_probs)
    num_modifs_per_step = [np.sum(num_step_choices == i) for i in range(max_num_steps)]
    ret = []
    for num_step_val_minus1, num_modifs_this_step in enumerate(num_modifs_per_step):
      if num_modifs_this_step == 0:
        continue
      new_nns = self.get_multi_step_modifications(nn, num_step_val_minus1 + 1,
                                                  num_modifs_this_step, **kwargs)
      ret.extend(new_nns)
    return ret

  @classmethod
  def _get_num_modifications(cls, list_of_opers, passed_num_modifs, dflt_num_modifs):
    """ Returns the number of modifications depending on the passed value. """
    if passed_num_modifs == 'all':
      return len(list_of_opers)
    elif passed_num_modifs is None:
      return dflt_num_modifs
    elif passed_num_modifs >= 0:
      return passed_num_modifs
    else:
      raise ValueError('num_*_step_modifications should be \'all\', ' +
                       'None or a positive integer. ')

  def _is_a_valid_network(self, nn):
    """ Returns true if it is a valid network. """
    if nn is None:
      return False
    elif self.constraint_checker is None:
      return True
    else:
      return self.constraint_checker(nn)

  def get_primitives_grouped_by_type(self, nn, types_of_primitives=None):
    """ Returns the list of primitives grouped by type. """
    #pylint: disable=bare-except
    # doing a bare except only for testing. Will remove soon.
    types_of_primitives = types_of_primitives if types_of_primitives is not None \
                          else _PRIMITIVE_PROB_MASSES.keys()
    ret = {}
    primitive_type_prob_masses = {}
    for top in types_of_primitives:
      top_is_of_known_type = True
      try:
        if top == 'inc_single':
          ret[top] = get_list_of_single_layer_modifiers(nn, 'increase',
                       self.options.spawn_single_inc_num_units,
                       self.options.single_inc_change_frac)
        elif top == 'dec_single':
          ret[top] = get_list_of_single_layer_modifiers(nn, 'decrease',
                       self.options.spawn_single_dec_num_units,
                       self.options.single_dec_change_frac)
        elif top == 'inc_en_masse':
          ret[top] = get_list_of_en_masse_change_primitives(nn, 'inc')
        elif top == 'dec_en_masse':
          ret[top] = get_list_of_en_masse_change_primitives(nn, 'dec')
        elif top == 'swap_layer':
          ret[top] = get_list_of_swap_layer_modifiers(nn)
        elif top == 'wedge_layer':
          ret[top] = get_list_of_wedge_layer_modifiers(nn)
        elif top == 'remove_layer':
          ret[top] = get_list_of_remove_layer_modifiers(nn)
        elif top == 'branch':
          ret[top] = get_list_of_branching_modifiers(nn)
        elif top == 'skip':
          ret[top] = get_list_of_skipping_modifiers(nn)
        else:
          top_is_of_known_type = False
      except:
        pass
      if not top_is_of_known_type:
        raise ValueError('All values in types_of_primitives should be in %s. Given %s'%(
                         str(_PRIMITIVE_PROB_MASSES.keys()), top))
      # Finally also add the probability mass
      primitive_type_prob_masses[top] = _PRIMITIVE_PROB_MASSES[top]
    return ret, primitive_type_prob_masses

  def get_single_step_modifications(self, nn, num_single_step_modifications='all',
                                    **kwargs):
    """ Returns a list of new neural networks which have undergone a single modification
        from nn. """
    prims_by_type, type_prob_masses = self.get_primitives_grouped_by_type(nn, **kwargs)
    groups = prims_by_type.keys()
    if num_single_step_modifications == 'all':
      prims_by_group = prims_by_type.values()
      modifiers = [elem for group_prims in prims_by_group for elem in group_prims]
      num_single_step_modifications = len(modifiers)
    else:
      prob_masses = np.array([type_prob_masses[key] for key in groups])
      prob_masses = prob_masses/prob_masses.sum()
      modif_groups = np.random.choice(groups, max(2*num_single_step_modifications, 20),
                                      p=prob_masses)
      # Shuffle each list
      for grp in groups:
        np.random.shuffle(prims_by_type[grp])
      # Now create a list of modifiers
      modifiers = []
      for grp in modif_groups:
        if len(prims_by_type[grp]) == 0:
          continue
        else:
          modifiers.append(prims_by_type[grp].pop())
    # Now go through each modifier and make changes.
    ret = []
    for modif in modifiers:
      curr_val = modif(nn)
      if self._is_a_valid_network(curr_val):
        ret.append(curr_val)
        if len(ret) >= num_single_step_modifications:
          break
    if len(ret) < 0.25 * num_single_step_modifications:
      self.reporter.writeln(('Consider reducing constraints. %d/%d modifications ' +
                             'accepted.')%(len(ret), num_single_step_modifications))
    return ret

  def get_multi_step_modifications(self, nn, num_steps, num_multi_step_modifications,
                                   **kwargs):
    """ Returns a list of new neural networks which have undergone num_steps
        modifications.
    """
    single_step_modifs = self.get_single_step_modifications(nn,
                           num_multi_step_modifications, **kwargs)
    if num_steps == 1:
      return single_step_modifs
    else:
      ret = []
      for new_nn in single_step_modifs:
        result = self.get_multi_step_modifications(new_nn, num_steps-1, 1, **kwargs)
        assert len(result) <= 1
        if len(result) > 0:
          ret.append(result[0])
        else:
          ret.append(new_nn)
      return ret


# An API to simply return an operation =================================================
def get_nn_modifier_from_args(constraint_checker, dflt_num_steps_probs=None,
                              dflt_max_num_steps=None, options=None, reporter=None):
  """ Returns a function which can be used to modify neural networks. """
  # A modifier which works with probabilities
  def _get_modifier_with_probs(_nn_modifier, _num_steps_probs):
    """ Returns a modifier with probs. """
    _num_steps_probs = np.array(_num_steps_probs)
    _num_steps_probs = _num_steps_probs/_num_steps_probs.sum()
    return lambda list_of_nns, num_modifications, **kwargs: _nn_modifier(list_of_nns,
                                      num_modifications, num_steps_probs=_num_steps_probs,
                                      max_num_steps=None, **kwargs)
  # A modifier which works with max_num_steps
  def _get_modifier_with_steps(_nn_modifier, _max_num_steps):
    """ Returns a modifier with max_num_steps. """
    return lambda list_of_nns, num_modifications, **kwargs: _nn_modifier(list_of_nns,
                                      num_modifications, num_steps_probs=None,
                                      max_num_steps=_max_num_steps, **kwargs)
  # Construct a Modifier
  nn_modifier = NNModifier(constraint_checker, options, reporter)
  if dflt_num_steps_probs is not None:
    return _get_modifier_with_probs(nn_modifier, dflt_num_steps_probs)
  elif dflt_max_num_steps is not None:
    return _get_modifier_with_steps(nn_modifier, dflt_max_num_steps)
  else:
    return nn_modifier

