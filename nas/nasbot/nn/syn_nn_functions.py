"""
  Implements various synthetic functions on NN architectures.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=invalid-name

import numpy as np

def _get_vals_wo_None(iter_of_vals):
  """ Returns a list of values without Nones. """
  return [x for x in iter_of_vals if x is not None]

def _num_units_signal(num_units_vals, bias_val, decay):
  """ Signal on the number of units. """
  num_units_vals = np.array(_get_vals_wo_None(num_units_vals))
  return np.exp(-decay * abs(num_units_vals.mean() - bias_val))

def _degree_signal(in_degrees, out_degrees, bias_val, decay):
  """ Signal on the degrees. """
  avg_degree = (in_degrees.mean() + out_degrees.mean())/2.0
  return np.exp(-decay * abs(avg_degree - bias_val))

def _get_ip_op_distance_signal(ip_op_dist, bias_val, decay):
  """ Signal on distance from input to output. """
  return np.exp(-decay * abs(ip_op_dist - bias_val))

def _get_layer_degree_signal(degree_of_layer, bias_val, decay):
  """ A signal based on the degree of a layer. """
  return np.exp(-decay * abs(degree_of_layer - bias_val))

def _get_num_layers_signal(num_layers, bias_val, decay):
  """ A signal based on the number of layers. """
  return np.exp(-decay * abs(num_layers - bias_val))

def _get_num_edges_signal(num_edges, bias_val, decay):
  """ A signal based on the total number of edges. """
  return np.exp(-decay * abs(num_edges - bias_val))

def _get_stride_signal(strides, bias_val, decay):
  """ A signal using the strides. """
  strides = np.array(_get_vals_wo_None(strides))
  return np.exp(-decay * abs(strides.mean() - bias_val))

def _get_conv_signal(layer_labels):
  """ A signal using the convolutional layers. """
  conv_layers = [ll for ll in layer_labels if \
                 ll.startswith('conv') or ll.startswith('res')]
  conv_filter_vals = np.array([float(ll[-1]) for ll in conv_layers])
  return (conv_filter_vals == 3).sum() / float(len(conv_filter_vals) + 1)

def _get_sigmoid_signal(layer_labels):
  """ A function using the sigmoid layer fraction as the signal. """
  internal_layers = [ll for ll in layer_labels if ll not in ['ip', 'op', 'linear']]
  good_layers = [ll in ['logistic', 'relu'] for ll in internal_layers]
  return sum(good_layers) / float(len(internal_layers) + 1)


def syn_func1_common(nn):
  """ A synthetic function on NN architectures. """
  return _num_units_signal(nn.num_units_in_each_layer, 1000, 0.002) + \
         _degree_signal(nn.get_in_degrees(), nn.get_out_degrees(), 5, 0.4) + \
         _get_ip_op_distance_signal(nn.get_distances_from_ip()[nn.get_op_layer_idx()],
                                    10, 0.2) + \
         _get_layer_degree_signal(nn.get_in_degrees()[nn.get_op_layer_idx()], 3, 0.5) + \
         _get_layer_degree_signal(nn.get_out_degrees()[nn.get_ip_layer_idx()], 4, 0.5) + \
         _get_num_layers_signal(nn.num_layers, 30, 0.1) + \
         _get_num_edges_signal(nn.conn_mat.sum(), 100, 0.05)


def cnn_syn_func1(nn):
  """ A synthetic function for CNNs. """
  return syn_func1_common(nn) + \
         _num_units_signal(nn.num_units_in_each_layer, 500, 0.001) + \
         _get_num_layers_signal(nn.num_layers, 50, 0.3) + \
         _get_stride_signal(nn.strides, 1.5, 3.0) + \
         _get_conv_signal(nn.layer_labels)


def mlp_syn_func1(nn):
  """ A synthetic function for MLPs. """
  return syn_func1_common(nn) + \
         _get_num_edges_signal(nn.conn_mat.sum(), 50, 0.1) + \
         _num_units_signal(nn.num_units_in_each_layer, 2000, 0.001) + \
         _get_sigmoid_signal(nn.layer_labels)

