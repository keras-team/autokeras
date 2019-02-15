"""
  Implements some example neural networks. We will mostly use them to instantiate the
  chains in GA and/or Simulated annealing.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=no-name-in-module

import numpy as np
from scipy.sparse import dok_matrix
# local imports
from nas.nasbot.nn.neural_network import ConvNeuralNetwork, MultiLayerPerceptron, \
                              get_cnn_layer_labels,\
                              get_mlp_layer_labels, is_a_conv_layer_label
from nas.nasbot.nn.nn_modifiers import NNModifier
from nas.nasbot.utils.general_utils import get_dok_mat_with_set_coords


# CNNs ===================================================================================
def get_feedforward_adj_mat(num_layers):
  """ Returns an adjacency matrix for a feed forward network. """
  ret = dok_matrix((num_layers, num_layers))
  for i in range(num_layers-1):
    ret[i, i+1] = 1
  return ret

def get_vgg_net(num_conv_layers_per_block=4, cnn_layer_labels=None):
  """ Returns a VGG net. """
  cnn_layer_labels = cnn_layer_labels if cnn_layer_labels is not None else \
                     get_cnn_layer_labels()
  layer_labels = ['ip', 'conv3', 'conv3', 'max-pool', 'conv3', 'conv3', 'max-pool']
  num_filters_each_layer = [None, 64, 64, None, 128, 128, None]
  # Now create the blocks
  block_filter_sizes = [128, 256, 512]
  for bfs in block_filter_sizes:
    layer_labels.extend(['conv3' for _ in range(num_conv_layers_per_block)] +
                        ['max-pool'])
    num_filters_each_layer.extend([bfs] * num_conv_layers_per_block + [None])
  layer_labels.extend(['fc', 'fc', 'fc', 'softmax', 'op'])
  num_filters_each_layer.extend([128, 256, 512, None, None])
  num_layers = len(layer_labels)
  # Construct the connectivity matrix
  conn_mat = get_feedforward_adj_mat(num_layers)
  strides = [(1 if is_a_conv_layer_label(ll) else None) for ll in layer_labels]
  vgg = ConvNeuralNetwork(layer_labels, conn_mat, num_filters_each_layer, strides,
                          cnn_layer_labels)
  return vgg


def _get_multidepth_cnn_eg12_common():
  """ A network with 2 softmax layers mostly for debugging common operations. """
  cnn_layer_labels = get_cnn_layer_labels()
  layer_labels = ['ip', 'op', 'softmax', 'fc', 'softmax', 'fc', 'conv5', 'avg-pool',
                  'max-pool', 'conv3', 'conv3', 'max-pool', 'max-pool', 'conv3', 'conv7']
  num_filters_each_layer = [None, None, None, 64, None, 64, 128, None, None, 64,
                            64, None, None, 128, 64]
  edges = [(0, 14), (14, 6), (14, 9), (14, 10), (6, 7), (7, 3), (3, 2), (2, 1), (9, 8),
           (8, 5), (5, 4), (4, 1), (10, 11), (11, 13), (13, 12), (12, 5)]
  strides = [(1 if is_a_conv_layer_label(ll) else None) for ll in layer_labels]
  return layer_labels, edges, num_filters_each_layer, cnn_layer_labels, strides


def get_multidepth_cnn_eg1():
  """ A network with 2 softmax layers mostly for debugging. """
  layer_labels, edges, num_filters_each_layer, cnn_layer_labels, strides = \
    _get_multidepth_cnn_eg12_common()
  edges.append((3, 4))
  conn_mat = get_dok_mat_with_set_coords(len(layer_labels), edges)
  strides[9] = 2
  return ConvNeuralNetwork(layer_labels, conn_mat, num_filters_each_layer, strides,
                           cnn_layer_labels)


def get_multidepth_cnn_eg2():
  """ A network with 2 softmax layers mostly for debugging. """
  layer_labels, edges, num_filters_each_layer, cnn_layer_labels, strides = \
    _get_multidepth_cnn_eg12_common()
  edges.append((6, 8))
  conn_mat = get_dok_mat_with_set_coords(len(layer_labels), edges)
  strides[9] = 2
  strides[6] = 2
  return ConvNeuralNetwork(layer_labels, conn_mat, num_filters_each_layer, strides,
                           cnn_layer_labels)


def _dflt_unit_sizes(num_blocks):
  """ Returns a default set of values for the unit sizes. """
  ret = []
  unit_size = 64
  for _ in range(num_blocks):
    ret.append(unit_size)
    unit_size = min(unit_size*2, 512)
  return ret


def _get_blocked_cnn_params(num_blocks, num_layers_per_block, block_layer_type,
  num_fc_layers, num_conv_filters_in_layers=None, num_fc_nodes_in_layers=None,
  cnn_layer_labels=None):
  """ Returns parameters for a blocked CNN.
    num_blocks: # blocks, i.e. the number of repeated convolutional layers.
    num_layers_per_block: # layers per block.
    num_fc_layers: # fully connected layers.
    num_conv_filters_in_layers: # filters in the layers for each block.
    num_fc_nodes_in_layers: sizes of the fc layers.
    cnn_layer_labels: Labels for all layer types in a CNN.
  """
  layer_labels = ['ip', 'conv7', 'max-pool']
  num_filters_each_layer = [None, 64, None]
  strides = [None, 1, None]
  num_conv_filters_in_layers = num_conv_filters_in_layers if num_conv_filters_in_layers \
    is not None else _dflt_unit_sizes(num_blocks)
  num_fc_nodes_in_layers = num_fc_nodes_in_layers if num_fc_nodes_in_layers is not None \
    else [2 * num_conv_filters_in_layers[-1]] * num_fc_layers
  cnn_layer_labels = cnn_layer_labels if cnn_layer_labels is not None else \
                     get_cnn_layer_labels()
  # Construct blocks
  for block_idx in range(num_blocks):
    layer_labels.extend([block_layer_type for _ in range(num_layers_per_block)])
    num_filters_each_layer.extend([num_conv_filters_in_layers[block_idx]] *
                         num_layers_per_block)
    strides.extend([2] + [1] * (num_layers_per_block - 1))
  # Pooling layer after the blocks
  layer_labels.append('avg-pool')
  num_filters_each_layer.append(None)
  strides.append(None)
  # Add FC layers
  layer_labels.extend(['fc' for _ in range(num_fc_layers)] + ['softmax', 'op'])
  num_filters_each_layer.extend(num_fc_nodes_in_layers + [None, None])
  strides.extend([None] * (num_fc_layers + 2))
  # Construct the connectivity matrix
  num_layers = len(layer_labels)
  conn_mat = get_feedforward_adj_mat(num_layers)
  return layer_labels, conn_mat, num_filters_each_layer, cnn_layer_labels, strides


def get_blocked_cnn(num_blocks, num_conv_layers_per_block, num_fc_layers,
                    num_conv_filters_in_layers=None,
                    num_fc_nodes_in_layers=None, cnn_layer_labels=None):
  """ Returns a blocked CNN.
    num_blocks: # blocks, i.e. the number of repeated convolutional layers.
    num_conv_layers_per_block: # convolutaional layers per block.
    num_fc_layers: # fully connected layers.
    num_conv_filters_in_layers: # filters in the layers for each block.
    num_fc_nodes_in_layers: sizes of the fc layers.
    cnn_layer_labels: Labels for all layer types in a CNN.
  """
  layer_labels, conn_mat, num_filters_each_layer, cnn_layer_labels, strides = \
    _get_blocked_cnn_params(num_blocks, num_conv_layers_per_block, 'conv3', num_fc_layers,
      num_conv_filters_in_layers, num_fc_nodes_in_layers, cnn_layer_labels)
  return ConvNeuralNetwork(layer_labels, conn_mat, num_filters_each_layer, strides,
                           cnn_layer_labels)


def get_resnet_cnn(num_res_blocks, num_conv_layers_per_block, num_fc_layers,
                   num_conv_filters_in_layers=None,
                   num_fc_nodes_in_layers=None, cnn_layer_labels=None):
  """ Returns a Resnet CNN.
      num_layers_to_skip: The number of layers to skip when adding skip connections.
      see get_blocked_cnn for other arguments.
  """
  layer_labels, conn_mat, num_filters_each_layer, cnn_layer_labels, strides = \
    _get_blocked_cnn_params(num_res_blocks, num_conv_layers_per_block, 'res3',
      num_fc_layers, num_conv_filters_in_layers, num_fc_nodes_in_layers, cnn_layer_labels)
  return ConvNeuralNetwork(layer_labels, conn_mat, num_filters_each_layer, strides,
                           cnn_layer_labels)


#MLPs ===================================================================================
def get_blocked_mlp(class_or_reg, num_blocks, num_layers_per_block=None,
                    num_units_in_each_layer=None):
  """ Creates a blocked MLP. """
  # Create rectifiers and sigmoids
  rectifiers = ['relu', 'elu', 'crelu', 'leaky-relu', 'softplus']
  sigmoids = ['logistic', 'tanh']
  obtl_label = 'linear' if class_or_reg == 'reg' else 'softmax'
  np.random.shuffle(rectifiers)
  np.random.shuffle(sigmoids)
  rect_count = 0
  sig_count = 0
  # Preprocess args
  # Create the network
  layer_labels = ['ip']
  num_units_in_each_layer = num_units_in_each_layer if num_units_in_each_layer \
                            is not None else _dflt_unit_sizes(num_blocks)
  # Construct blocks
  for block_idx in range(num_blocks):
    if block_idx % 2 == 0:
      layer_type = rectifiers[rect_count%len(rectifiers)]
      rect_count += 1
    else:
      layer_type = sigmoids[sig_count%len(sigmoids)]
      sig_count += 1
    layer_labels.extend([layer_type] * num_layers_per_block)
    num_units_in_each_layer.extend([num_units_in_each_layer[block_idx]] *
                                   num_layers_per_block)
  # A linear layer at the end of the block
  layer_labels.extend([obtl_label, 'op'])
  num_units_in_each_layer.extend([None, None])
  conn_mat = get_feedforward_adj_mat(len(layer_labels))
  all_layer_labels = get_mlp_layer_labels(class_or_reg)
  return MultiLayerPerceptron(class_or_reg, layer_labels, conn_mat,
                              num_units_in_each_layer, all_layer_labels)


def _get_multidepth_mlp_eg12_common(class_or_reg):
  """ A network with 2 linear layers mostly for debugging common operations. """
  mlp_layer_labels = get_mlp_layer_labels(class_or_reg)
  obtl_label = 'linear' if class_or_reg == 'reg' else 'softmax'
  layer_labels = ['ip', obtl_label, 'op', 'tanh', 'relu', 'leaky-relu', 'logistic',
                  'logistic', 'elu', obtl_label]
  num_units_in_each_layer = [None, None, None, 64, 64, 128, 256, 64, 512, None]
  edges = [(0, 1), (0, 3), (0, 4), (3, 5), (4, 7), (5, 6), (7, 8), (3, 8), (7, 6), \
           (6, 9), (8, 9), (1, 2), (9, 2)]
  return layer_labels, num_units_in_each_layer, edges, mlp_layer_labels

def get_multidepth_mlp_eg1(class_or_reg):
  """ Multi depth MLP eg 1. """
  layer_labels, num_units_in_each_layer, edges, mlp_layer_labels = \
    _get_multidepth_mlp_eg12_common(class_or_reg)
  conn_mat = get_dok_mat_with_set_coords(len(layer_labels), edges)
  return MultiLayerPerceptron(class_or_reg, layer_labels, conn_mat,
                              num_units_in_each_layer, mlp_layer_labels)

def get_multidepth_mlp_eg2(class_or_reg):
  """ Multi depth MLP eg 2. """
  layer_labels, num_units_in_each_layer, edges, mlp_layer_labels = \
    _get_multidepth_mlp_eg12_common(class_or_reg)
  edges.append((4, 1))
  conn_mat = get_dok_mat_with_set_coords(len(layer_labels), edges)
  return MultiLayerPerceptron(class_or_reg, layer_labels, conn_mat,
                              num_units_in_each_layer, mlp_layer_labels)


# Some functions to generate neural network architectures ---------------------------
def generate_cnn_architectures():
  # pylint: disable=bad-whitespace
  """ Generates 4 neural networks. """
  all_layer_label_classes = get_cnn_layer_labels()
  # Network 1
  layer_labels = ['ip', 'op', 'conv3', 'fc', 'conv3', 'conv3', 'conv3', 'softmax',
                  'max-pool']
  num_filters_each_layer = [None, None, 32, 16, 16, 8, 8, None, None]
  A = get_dok_mat_with_set_coords(9,
    [(0,4), (3,7), (4,5), (4,6), (5,2), (6,2), (7,1), (2, 8), (8, 3)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  cnn_1 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 2
  layer_labels = ['ip', 'conv3', 'conv3', 'conv3', 'fc', 'softmax', 'op', 'max-pool']
  num_filters_each_layer = [None, 16, 16, 32, 16, None, None, None]
  A = get_dok_mat_with_set_coords(8, [(0,1), (1,2), (2,3), (4,5), (5,6), (3, 7),
                                      (7, 4)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  cnn_2 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 3
  layer_labels = ['ip', 'conv3', 'conv3', 'conv5', 'conv3', 'max-pool',
                  'fc', 'softmax', 'op']
  num_filters_each_layer = [None, 16, 16, 16, 32, None, 32, None, None]
  A = get_dok_mat_with_set_coords(9, [(0,1), (1,2), (1,3), (1,4), (2,5), (3,5), (4,6),
                                       (5,6), (6,7), (7,8)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  strides[4] = 2
  cnn_3 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 4
  layer_labels = ['ip', 'conv3', 'conv3', 'conv5', 'conv3', 'avg-pool', 'conv5', 'fc',
                  'softmax', 'op']
  num_filters_each_layer = [None, 16, 16, 16, 32, None, 32, 32, None, None]
  A = get_dok_mat_with_set_coords(10, [(0,1), (1,2), (1,3), (1,4), (2,5), (3,5), (4,6),
                                        (5,7), (6,7), (7,8), (8,9)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool', 'avg-pool']
             else 1 for ll in layer_labels]
  strides[4] = 2
  cnn_4 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 5
  layer_labels = ['ip', 'conv3', 'conv3', 'conv5', 'conv5', 'avg-pool',
                  'fc', 'softmax', 'op', 'conv3']
  num_filters_each_layer = [None, 16, 16, 16, 32, None, 32, None, None, 16]
  A = get_dok_mat_with_set_coords(10, [(0,1), (1,2), (1,3), (2,5), (3,5), (4,6),
                                       (5,6), (6,7), (7,8), (0, 9), (9, 4)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'avg-pool']
             else 1 for ll in layer_labels]
  strides[4] = 2
  cnn_5 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 6
  layer_labels = ['ip', 'conv3', 'conv3', 'conv3', 'fc', 'fc', 'op', 'max-pool',
                  'fc', 'softmax']
  num_filters_each_layer = [None, 16, 16, 32, 32, 32, None, None, 32, None]
  A = get_dok_mat_with_set_coords(10, [(0,1), (1,2), (2,3), (4,5), (5,9), (3,7),
                                      (7,4), (4,8), (8,9), (9,6)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  cnn_6 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 7 - Two softmax layers both pointing to output
  layer_labels = ['ip', 'conv3', 'conv3', 'conv3', 'fc', 'fc', 'op', 'max-pool',
                  'fc', 'softmax', 'softmax']
  num_filters_each_layer = [None, 16, 16, 32, 32, 32, None, None, 32, None, None]
  A = get_dok_mat_with_set_coords(11, [(0,1), (1,2), (2,3), (4,5), (5,9), (3,7),
                                      (7,4), (4,8), (8,9), (9,6), (8,10), (10,6)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  cnn_7 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 8 - Similar to previous, except with residual layers
  layer_labels = ['ip', 'conv3', 'res3', 'res5', 'fc', 'fc', 'op', 'max-pool',
                  'fc', 'softmax', 'softmax']
  num_filters_each_layer = [None, 16, 16, 32, 32, 32, None, None, 32, None, None]
  A = get_dok_mat_with_set_coords(11, [(0,1), (1,2), (2,3), (4,5), (5,9), (3,7),
                                      (7,4), (4,8), (8,9), (9,6), (8,10), (10,6)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  cnn_8 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)
  # Network 9 - Similar to previous, except decreasing units for residual layers
  layer_labels = ['ip', 'conv3', 'res3', 'res5', 'res7', 'fc', 'op', 'max-pool',
                  'fc', 'softmax', 'softmax']
  num_filters_each_layer = [None, 16, 32, 8, 32, 128, None, None, 256, None, None]
  A = get_dok_mat_with_set_coords(11, [(0,1), (1,2), (2,3), (4,5), (5,9), (3,7),
                                      (7,4), (4,8), (8,9), (9,6), (8,10), (10,6)])
  strides = [None if ll in ['ip', 'op', 'fc', 'softmax', 'max-pool']
             else 1 for ll in layer_labels]
  cnn_9 = ConvNeuralNetwork(layer_labels, A,
            num_filters_each_layer, strides, all_layer_label_classes)

  return [cnn_1, cnn_2, cnn_3, cnn_4, cnn_5, cnn_6, cnn_7, cnn_8, cnn_9,
          get_vgg_net(2),
          get_blocked_cnn(3, 4, 1),
          get_resnet_cnn(3, 2, 1),
          get_multidepth_cnn_eg1(),
          get_multidepth_cnn_eg2(),
         ]


def generate_mlp_architectures(class_or_reg='reg'):
  """ pylint: disable=bad-whitespace. """
  # pylint: disable=bad-whitespace
  all_layer_label_classes = get_mlp_layer_labels(class_or_reg)
  last_layer_label = 'linear' if class_or_reg == 'reg' else 'softmax'
  # Network 1
  layer_labels = ['ip', 'op', 'tanh', 'logistic', 'softplus', 'relu', 'elu',
                  last_layer_label]
  num_units_each_layer = [None, None, 32, 64, 16, 8, 8, None]
  A = get_dok_mat_with_set_coords(8,
    [(0,4), (2,3), (3,7), (4,5), (4,6), (5,2), (6,2), (7,1)])
  mlp_1 = MultiLayerPerceptron(class_or_reg, layer_labels, A,
            num_units_each_layer, all_layer_label_classes)
  # Network 2
  layer_labels = ['ip', 'softplus', 'elu', 'tanh', 'logistic', last_layer_label, 'op']
  num_units_each_layer = [None, 16, 16, 32, 64, None, None]
  A = get_dok_mat_with_set_coords(7, [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6)])
  mlp_2 = MultiLayerPerceptron(class_or_reg, layer_labels, A,
            num_units_each_layer, all_layer_label_classes)
  # Network 3
  layer_labels = ['ip', 'tanh', 'logistic', 'logistic', 'tanh', 'elu', 'relu',
                  last_layer_label, 'op']
  num_units_each_layer = [None, 8, 8, 8, 8, 16, 16, None, None]
  A = get_dok_mat_with_set_coords(9, [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (3, 6),
                                      (4, 5), (4, 6), (5, 7), (6, 7), (7, 8)])
  mlp_3 = MultiLayerPerceptron(class_or_reg, layer_labels, A,
            num_units_each_layer, all_layer_label_classes)
  # Network 4
  layer_labels = ['ip', 'logistic', 'relu', 'softplus', last_layer_label, 'op', 'relu',
                  'tanh']
  num_units_each_layer = [None, 8, 8, 16, None, None, 16, 8]
  A = get_dok_mat_with_set_coords(8, [(0, 1), (0, 7), (1, 2), (2, 3), (2, 6), (7, 6),
                                      (7, 3), (3, 4), (6, 4), (4, 5)])
  mlp_4 = MultiLayerPerceptron(class_or_reg, layer_labels, A,
            num_units_each_layer, all_layer_label_classes)
  return [mlp_1, mlp_2, mlp_3, mlp_4,
          get_blocked_mlp(class_or_reg, 4, 3),
          get_blocked_mlp(class_or_reg, 8, 2),
          get_multidepth_mlp_eg1(class_or_reg),
          get_multidepth_mlp_eg2(class_or_reg),
         ]


def generate_many_neural_networks(nn_type, num_nns=200):
  """ Generates a large number of MLPs. """
  if nn_type == 'cnn':
    initial_nns = generate_cnn_architectures()
  elif nn_type == 'mlp-reg':
    initial_nns = generate_mlp_architectures('reg')
  elif nn_type == 'mlp-class':
    initial_nns = generate_mlp_architectures('class')
  else:
    raise ValueError('Unknown nn_type: %s.'%(nn_type))
  num_steps_probs = np.array([0.0, 0.0, 0.1, 0.2, 0.3, 0.4])
  num_steps_probs = num_steps_probs / num_steps_probs.sum()
  num_iters = 5
  num_nns_per_iter = int(num_nns / float(num_iters))
  # Get modifier
  modifier = NNModifier(None)
  curr_nns = initial_nns
  ret = [elem for elem in initial_nns]
  while len(ret) <= num_nns:
    curr_nns = modifier(curr_nns, num_nns_per_iter, num_steps_probs)
    ret.extend(curr_nns)
  np.random.shuffle(ret)
  return ret[:num_nns]

