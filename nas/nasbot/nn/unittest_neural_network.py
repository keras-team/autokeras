"""
  Test cases for functions/classes in neural_network.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name

import numpy as np
# Local imports
from nas.nasbot.nn import neural_network
from nas.nasbot.nn.nn_examples import generate_cnn_architectures, generate_mlp_architectures
from nas.nasbot.utils.base_test_class import BaseTestClass, execute_tests
from nas.nasbot.utils.graph_utils import apsp_floyd_warshall_costs


# Test cases for NeuralNetwork class -------------------------------------------------
class NeuralNetworkTestCase(BaseTestClass):
  """ Unit tests for the NeuralNetwork test class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NeuralNetworkTestCase, self).__init__(*args, **kwargs)
    self.cnns = generate_cnn_architectures()
    self.mlps_reg = generate_mlp_architectures('reg')
    self.mlps_class = generate_mlp_architectures('class')

  def _print_path_lengths(self, nns, fwd_or_bkwd, path_length_types):
    """ Prints the path length types. """
    # pylint: disable=protected-access
    # disabling this because this property should be private but printing this out during
    # a test might help catch some bugs.
    for idx, nn in enumerate(nns):
      self.report('%s-%d:: #layers=%d, #edges=%d.'%(nn.nn_class, idx, nn.num_layers,
                                                    nn.conn_mat.sum()), 'test_result')
      if fwd_or_bkwd == 'fwd':
        curr_path_lengths = nn._fwd_dists_to_op
      elif fwd_or_bkwd == 'bkwd':
        curr_path_lengths = nn._bkwd_dists_to_ip
      else:
        curr_path_lengths = None
      # Print them out
      for plt in path_length_types:
        self.report('  %s[%s]: %s.'%(fwd_or_bkwd, plt, curr_path_lengths[plt]),
                    'test_result')

  def test_cnn_instantiation(self):
    """ Tests instantiation. """
    self.report('Testing instantiation of cnns.')
    path_length_types = ['all-shortest', 'all-longest', 'all-rw',
                         'conv-shortest', 'conv-longest', 'conv-rw',
                         'pool-shortest', 'pool-longest', 'pool-rw',
                         'fc-shortest', 'fc-longest', 'fc-rw',
                        ]
    self._print_path_lengths(self.cnns, 'bkwd', path_length_types)
    self._print_path_lengths(self.cnns, 'fwd', path_length_types)

  def test_mlp_instantiation(self):
    """ Tests instantiation. """
    path_length_types = ['all-shortest', 'all-longest', 'all-rw',
                         'sigmoid-shortest', 'sigmoid-longest', 'sigmoid-rw',
                         'rectifier-shortest', 'rectifier-longest', 'rectifier-rw',
                        ]
    self.report('Testing instantiation of mlps - regression.')
    self._print_path_lengths(self.mlps_reg, 'bkwd', path_length_types)
    self._print_path_lengths(self.mlps_reg, 'fwd', path_length_types)
    self.report('Testing instantiation of mlps - classification.')
    self._print_path_lengths(self.mlps_class, 'bkwd', path_length_types)
    self._print_path_lengths(self.mlps_class, 'fwd', path_length_types)

  def test_bkwd_ip_fwd_op_dists_of_all_layers(self):
    """ Tests computation and retrieval of the distances to all layers. """
    self.report('Testing computation/retrieval of backward and forward distances.')
    for cnn in self.cnns:
      all_label_order = ['all-shortest', 'conv-shortest', 'pool-shortest']
      np.random.shuffle(all_label_order)
      # First comptue the distances and check if the sizes are ok.
      curr_bkwd_to_ip_comp, curr_fwd_to_op_comp = \
        cnn.get_bkwd_ip_fwd_op_dists_of_all_layers(all_label_order)
      assert curr_fwd_to_op_comp.shape == (cnn.num_layers, len(all_label_order))
      assert curr_bkwd_to_ip_comp.shape == (cnn.num_layers, len(all_label_order))
      # Now check for values - we will use floyd warshall to compute the distances here.
      ip_idx = cnn.get_ip_layer_idx()
      op_idx = cnn.get_op_layer_idx()
      curr_edge_wts = cnn.get_edge_weights_from_conn_mat()
      curr_fwd_to_op = []
      curr_bkwd_to_ip = []
      # compute the all pairs distances
      for ll in all_label_order:
        if ll == 'all-shortest':
          curr_fwd_to_op.append(
            apsp_floyd_warshall_costs(curr_edge_wts)[:, op_idx])
          curr_bkwd_to_ip.append(
            apsp_floyd_warshall_costs(curr_edge_wts.T)[:, ip_idx])
        else:
          dist_type = ll.split('-')[0]
          ll_edge_wts = cnn.get_layer_or_group_edge_weights_from_edge_weights(
                          curr_edge_wts.T, dist_type)
          ll_edge_wtsT = cnn.get_layer_or_group_edge_weights_from_edge_weights(
                           curr_edge_wts, dist_type)
          curr_bkwd_to_ip.append(
            apsp_floyd_warshall_costs(ll_edge_wtsT)[ip_idx, :])
          curr_fwd_to_op.append(
            apsp_floyd_warshall_costs(ll_edge_wts)[op_idx, :])
      curr_fwd_to_op = np.array(curr_fwd_to_op).T
      curr_bkwd_to_ip = np.array(curr_bkwd_to_ip).T
#       assert np.all(curr_bkwd_to_ip == curr_bkwd_to_ip_comp)
#       assert np.all(curr_fwd_to_op == curr_fwd_to_op_comp)


# Test cases for general NN functions -------------------------------------------------
class GeneralNNFunctionsTestCase(BaseTestClass):
  """ Unit tests for general NN functions. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(GeneralNNFunctionsTestCase, self).__init__(*args, **kwargs)

  def test_get_labels(self):
    """ Tests labels generated for a CNN and MLP. """
    self.report('Testing the labels for CNN and MLP.')
    cnn_true_labels_7 = sorted(['ip', 'op', 'fc', 'max-pool', 'avg-pool', 'softmax',
                                'res3', 'res5', 'res7', 'conv3', 'conv5', 'conv7'])
    cnn_true_labels_3 = sorted(['ip', 'op', 'fc', 'max-pool', 'avg-pool', 'softmax',
                                'res3', 'conv3'])
    mlp_true_labels = sorted(['ip', 'op', 'relu', 'linear'])
    cnn_ret_labels_7 = sorted(neural_network.get_cnn_layer_labels(7))
    cnn_ret_labels_3 = sorted(neural_network.get_cnn_layer_labels(3))
    mlp_ret_labels = sorted(neural_network.get_mlp_layer_labels('reg', ['relu']))
    assert cnn_true_labels_7 == cnn_ret_labels_7
    assert cnn_true_labels_3 == cnn_ret_labels_3
    assert mlp_true_labels == mlp_ret_labels


if __name__ == '__main__':
  execute_tests()

