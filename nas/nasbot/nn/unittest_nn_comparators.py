"""
  Test cases for functions/classes in nn_comparator.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name

import numpy as np
# Local imports
from ..utils.ancillary_utils import get_list_of_floats_as_str
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..nn import nn_comparators
from ..nn.unittest_neural_network import generate_cnn_architectures, \
                                       generate_mlp_architectures

_TOL = 1e-5


class TransportNNDistanceComputerTestCase(BaseTestClass):
  """ Contains unit tests for the TransportNNDistanceComputer class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(TransportNNDistanceComputerTestCase, self).__init__(*args, **kwargs)
    self.non_assignment_penalty = 1
    cnn_layer_labels, label_mismatch_penalty = \
      nn_comparators.get_cnn_layer_label_mismatch_penalties(self.non_assignment_penalty)
    self.tp_comp = nn_comparators.OTMANNDistanceComputer(cnn_layer_labels,
                     label_mismatch_penalty, self.non_assignment_penalty,
                     nn_comparators.CNN_STRUCTURAL_PENALTY_GROUPS,
                     nn_comparators.PATH_LENGTH_TYPES,
                     dflt_mislabel_coeffs=1.0, dflt_struct_coeffs=1.0)
    self.cnns = generate_cnn_architectures()

  def test_cnn_label_mismatch_penalties(self):
    """ Unit test for the label mismatch penalty of a CNN. """
    self.report('Testing generation of label mismatches for a CNN. ')
    cnn_layer_labels, label_mismatch_penalty = \
      nn_comparators.get_cnn_layer_label_mismatch_penalties(self.non_assignment_penalty,
                       max_conv_size=9)
    self.report('cnn_layer_labels: %s'%(str(cnn_layer_labels)), 'test_result')
    self.report('cnn mismatch penalties: \n%s'%(str(np.round(label_mismatch_penalty, 3))),
                                            'test_result')
    assert np.all(label_mismatch_penalty == label_mismatch_penalty.T)
    assert np.all(np.diag(label_mismatch_penalty) == 0)

  def test_mlp_label_mismatch_penalties(self):
    """ Unit test for the label mismatch penalty of an MLP. """
    self.report('Testing generation of label mismatches for a MLP. ')
    mlp_layer_labels, label_mismatch_penalty = \
      nn_comparators.get_mlp_layer_label_mismatch_penalties(self.non_assignment_penalty,
                                                            'reg')
    self.report('mlp_layer_labels: %s'%((str(mlp_layer_labels))), 'test_result')
    self.report('mlp mismatch penalties: \n%s'%(str(np.round(label_mismatch_penalty, 3))),
                'test_result')
    assert np.all(label_mismatch_penalty == label_mismatch_penalty.T)
    assert np.all(np.diag(label_mismatch_penalty) == 0)

  @classmethod
  def _is_cost_matrix_for_same_networks(cls, cost_matrix):
    """ Returns true if it is the cost matrix for the same network. """
    return np.all(np.diag(cost_matrix) == 0) and np.all(cost_matrix == cost_matrix.T)

  @classmethod
  def _has_corresponding_layers(cls, cost_matrix):
    """ Returns true if one network has a corresponding layer in the other and vice
        versa. """
    ret = True
    for row_idx in range(cost_matrix.shape[0]):
      ret = ret and np.any(cost_matrix[row_idx, :] == 0)
    for col_idx in range(cost_matrix.shape[1]):
      ret = ret and np.any(cost_matrix[:, col_idx] == 0)
    return ret

  def test_mislabel_cost_matrix(self):
    """ Tests the mislabel cost matrix for specific pairs of neural networks. """
    self.report('Testing generation of label cost matrices for specific cnns. ')
    num_cnns = len(self.cnns)
    for i in range(num_cnns):
      for j in range(i+1, num_cnns):
        cnn_i = self.cnns[i]
        cnn_j = self.cnns[j]
        mislabel_cost_matrix = self.tp_comp.get_mislabel_cost_matrix(cnn_i, cnn_j)
        assert mislabel_cost_matrix.shape[0] == cnn_i.num_layers
        assert mislabel_cost_matrix.shape[1] == cnn_j.num_layers
        if i == j:
          assert self._is_cost_matrix_for_same_networks(mislabel_cost_matrix)
        if i == 0 and j == 1:
          # These two matrices were designed so that there has to be a zero for each
          # column on each row and vice versa.
          assert self._has_corresponding_layers(mislabel_cost_matrix)
        if (i == 2 and j == 3) or (i == 0 and j == 1) or (i == 1 and j == 6):
          self.report('Mislabel cost matrix for cnn-%d and cnn-%d:\n%s'%(i, j,
                      str(np.round(mislabel_cost_matrix, 3))), 'test_result')

  def test_connectivity_cost_matrix(self):
    """ Tests the connectivity cost matrix for specific pairs of neural networks.  """
    self.report('Testing generation of connectivity cost matrices for specific cnns.')
    num_cnns = len(self.cnns)
    for i in range(num_cnns):
      for j in range(i, num_cnns):
        cnn_i = self.cnns[i]
        cnn_j = self.cnns[j]
        struct_cost_matrix = self.tp_comp.get_struct_cost_matrix(cnn_i, cnn_j)
        assert struct_cost_matrix.shape[0] == cnn_i.num_layers
        assert struct_cost_matrix.shape[1] == cnn_j.num_layers
        if i == j:
          assert self._is_cost_matrix_for_same_networks(struct_cost_matrix)
        if i == 0 and j == 1:
          # These two matrices were designed so that there has to be a zero for each
          # column on each row and vice versa.
          assert self._has_corresponding_layers(struct_cost_matrix)
        if (i == 2 and j == 3) or (i == 0 and j == 1) or (i == 1 and j == 6):
          self.report('Structural cost matrix for cnn-%d and cnn-%d:\n%s'%(i, j,
                      str(np.round(struct_cost_matrix, 3))), 'test_result')

  def test_ot_cost_matrix(self):
    """ Tests the OT cost matrix for specific pairs of neural networks.  """
    self.report('Testing generation of OT cost matrices for specific cnns.')
    nns = self.cnns
    num_nns = len(nns)
    for i in range(num_nns):
      for j in range(i, num_nns):
        nn_i = nns[i]
        nn_j = nns[j]
        mislabel_cost_matrix = self.tp_comp.get_mislabel_cost_matrix(nn_i, nn_j)
        struct_cost_matrix = self.tp_comp.get_struct_cost_matrix(nn_i, nn_j)
        ot_cost_matrix = self.tp_comp.get_ot_cost_matrix(mislabel_cost_matrix,
                           struct_cost_matrix, 1, 0.1, 1, None)
        assert ot_cost_matrix.shape[0] == nn_i.num_layers + 1
        assert ot_cost_matrix.shape[1] == nn_j.num_layers + 1
        if i == j:
          assert self._is_cost_matrix_for_same_networks(ot_cost_matrix)
        if i == 0 and j == 1:
          # These two matrices were designed so that there has to be a zero for each
          # column on each row and vice versa.
          assert self._has_corresponding_layers(ot_cost_matrix)
        if (i == 2 and j == 3) or (i == 0 and j == 1) or (i == 1 and j == 6):
          self.report('OT cost matrix for cnn-%d and cnn-%d:\n%s'%(i, j,
                      str(np.round(ot_cost_matrix, 3))), 'test_result')

  @classmethod
  def _get_dist_type_abbr(cls, dist_type):
    """ Shortens distance type. """
    if dist_type == 'lp-norm-by-max':
      return 'lnbm'
    else:
      return dist_type

  def _test_dist_comp_for_single_conn_coeff(self, nns, dist_types, tp_comp):
    """ Tests distance computation for a single connectivity coefficient. """
    num_nns = len(nns)
    for i in range(num_nns):
      for j in range(i, num_nns):
        nn_i = nns[i]
        nn_j = nns[j]
        dists = {}
        for dt in dist_types:
          dists[dt] = tp_comp(nn_i, nn_j, dist_type=dt)
        res_str = ' '.join(['%s=%s'%(self._get_dist_type_abbr(key),
                                     get_list_of_floats_as_str(val))
                            for key, val in dists.iteritems()])
        self.report('(i,j)=(%d,%d) %s'%(i, j, res_str), 'test_result')

  def _test_dist_comp_for_multiple_coeffs(self, nns, dist_types, mislabel_coeffs,
                                          struct_coeffs, tp_comp):
    """ Tests distance computation for a single connectivity coefficient. """
    num_nns = len(nns)
    for i in range(num_nns):
      for j in range(i, num_nns):
        nn_i = nns[i]
        nn_j = nns[j]
        dists = {}
        for dt in dist_types:
          dists[dt] = tp_comp(nn_i, nn_j, dist_type=dt, mislabel_coeffs=mislabel_coeffs,
                              struct_coeffs=struct_coeffs)
          num_dists = len(dt.split('-')) * len(mislabel_coeffs)
          assert len(dists[dt]) == num_dists
        res_str = ' '.join(['%s=%s'%(self._get_dist_type_abbr(key),
                                     get_list_of_floats_as_str(val))
                            for key, val in dists.iteritems()])
        self.report('(i,j)=(%d,%d) %s'%(i, j, res_str), 'test_result')

  def test_cnn_distance_computation(self):
    """ Tests the computation of the distance for CNNs. """
    struct_coeffs = [0.1, 0.4]
    mislabel_coeffs = [1.0] * len(struct_coeffs)
    dist_types = ['lp-emd', 'lp', 'emd']
    self.report('Testing distance computation for specific cnns with default coeffs.')
    self._test_dist_comp_for_single_conn_coeff(self.cnns, dist_types, self.tp_comp)
    # With multiple conn_coeffs
    self.report('Testing distance computation for specific cnns with conn_coeffs=%s.'%(
                struct_coeffs))
    self._test_dist_comp_for_multiple_coeffs(self.cnns, dist_types, mislabel_coeffs,
                                             struct_coeffs, self.tp_comp)

  def test_mlp_distance_computation(self):
    """ Tests the computation of the distance for CNNs. """
    self.report('Testing distance computation for specific mlps.')
    # Create the transport computation object
    mlp_layer_labels, label_mismatch_penalty = \
      nn_comparators.get_mlp_layer_label_mismatch_penalties(self.non_assignment_penalty,
                                                            'reg')
    mlp_tp_comp = nn_comparators.OTMANNDistanceComputer(mlp_layer_labels,
                     label_mismatch_penalty,
                     self.non_assignment_penalty,
                     nn_comparators.MLP_STRUCTURAL_PENALTY_GROUPS,
                     nn_comparators.PATH_LENGTH_TYPES,
                     dflt_mislabel_coeffs=1.0, dflt_struct_coeffs=1.0)
    # Create the mlp architectures
    mlps = generate_mlp_architectures()
    struct_coeffs = [0.1, 0.2]
    dist_types = ['lp-emd', 'lp', 'emd']
    mislabel_coeffs = [1.0] * len(struct_coeffs)
    self.report('Testing distance computation for specific mlps with default coeffs.')
    self._test_dist_comp_for_single_conn_coeff(mlps, dist_types, mlp_tp_comp)
    self.report('Testing distance computation for specific mlps with conn_coeffs=%s.'%(
                struct_coeffs))
    self._test_dist_comp_for_multiple_coeffs(mlps, dist_types, mislabel_coeffs,
                                             struct_coeffs, mlp_tp_comp)


class DistProdNNKernelTestCase(BaseTestClass):
  """ Unit tests for the Transport NNKernels. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(DistProdNNKernelTestCase, self).__init__(*args, **kwargs)
    self.non_assignment_penalty = 1
    cnn_layer_labels, label_mismatch_penalty = \
      nn_comparators.get_cnn_layer_label_mismatch_penalties(self.non_assignment_penalty)
    self.all_layer_labels = cnn_layer_labels
    self.label_mismatch_penalty = label_mismatch_penalty
    self.tp_comp = nn_comparators.OTMANNDistanceComputer(cnn_layer_labels,
                     label_mismatch_penalty,
                     self.non_assignment_penalty,
                     nn_comparators.CNN_STRUCTURAL_PENALTY_GROUPS,
                     nn_comparators.PATH_LENGTH_TYPES
                     )
    self.mislabel_coeffs = [2.0, 2.0, 1.0, 1.0, 1.0]
    self.struct_coeffs = [0.25, 0.5, 1.0, 2.0, 4.0]
    self.lp_betas = [1e-6] * len(self.struct_coeffs)
    self.emd_betas = [1] * len(self.struct_coeffs)
    self.scale = 1
    self.cnns = generate_cnn_architectures()

  def test_instantiation(self):
    """ Testing instantiation. """
    self.report('Testing instantiation of DistProdNNKernelTestCase and computation ' +
                'for specific networks.')
    dist_type_vals = ['lp', 'emd', 'lp-emd']
    all_kernels = []
    for dist_type in dist_type_vals:
      if dist_type == 'lp':
        betas = self.lp_betas
      elif dist_type == 'emd':
        betas = self.emd_betas
      else:
        betas = [j for i in zip(self.lp_betas, self.emd_betas) for j in i]
      tp_kernel = nn_comparators.generate_otmann_kernel_from_params('prod',
                    self.all_layer_labels, self.label_mismatch_penalty,
                    self.non_assignment_penalty,
                    nn_comparators.CNN_STRUCTURAL_PENALTY_GROUPS,
                    nn_comparators.PATH_LENGTH_TYPES,
                    self.mislabel_coeffs, self.struct_coeffs, betas,
                    self.scale, dist_type=dist_type)
      cnn_K = tp_kernel(self.cnns)
      all_kernels.append(cnn_K)
      cnn_eig_vals, _ = np.linalg.eig(cnn_K)
      self.report('dist-type: %s, eigvals: %s.'%(dist_type,
                     get_list_of_floats_as_str(sorted(cnn_eig_vals))))
      self.report('%s transport kernel:\n%s'%(dist_type,
                  str(np.round(cnn_K, 3))), 'test_result')
      assert cnn_K.shape == (len(self.cnns), len(self.cnns))
      assert np.all(np.diag(cnn_K) == 1)
    # Check if it is in fact the product
    if 'lp' in dist_type_vals and 'emd' in dist_type_vals and 'lp-emd' in dist_type_vals:
      lp_kernel = all_kernels[dist_type_vals.index('lp')]
      emd_kernel = all_kernels[dist_type_vals.index('emd')]
      lpemd_kernel = all_kernels[dist_type_vals.index('lp-emd')]
      assert np.linalg.norm(lpemd_kernel - lp_kernel * emd_kernel) < _TOL

  def test_kernel_computation(self):
    """ Testing computation of the lp distance. """
    self.report('Testing computed kernel values for specific cnns.')
    betas = [0.0001]
    scale = 2.1
    struct_coeffs = 1.0
    mislabel_coeffs = 1.0
    tp_comp = nn_comparators.OTMANNDistanceComputer(self.all_layer_labels,
                self.label_mismatch_penalty, self.non_assignment_penalty,
                nn_comparators.CNN_STRUCTURAL_PENALTY_GROUPS,
                nn_comparators.PATH_LENGTH_TYPES,
                dflt_mislabel_coeffs=mislabel_coeffs,
                dflt_struct_coeffs=struct_coeffs)
    tp_kernel = nn_comparators.generate_otmann_kernel_from_params('prod',
                self.all_layer_labels, self.label_mismatch_penalty,
                self.non_assignment_penalty, nn_comparators.CNN_STRUCTURAL_PENALTY_GROUPS,
                nn_comparators.PATH_LENGTH_TYPES,
                mislabel_coeffs, struct_coeffs, betas, scale,
                dist_type='lp')
    cnn_dists = tp_comp(self.cnns, self.cnns)
    cnn_kernel = tp_kernel(self.cnns)
    diff = np.linalg.norm(scale * np.exp(-cnn_dists[0] * betas[0]) - cnn_kernel)
    assert diff < np.linalg.norm(cnn_kernel) * 1e-6


class DistSumNNKernelTestCase(BaseTestClass):
  """ Unit tests for the Transport NNKernels. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(DistSumNNKernelTestCase, self).__init__(*args, **kwargs)
    self.non_assignment_penalty = 1
    mlp_layer_labels, label_mismatch_penalty = \
      nn_comparators.get_mlp_layer_label_mismatch_penalties(self.non_assignment_penalty,
                                                            'reg')
    self.all_layer_labels = mlp_layer_labels
    self.label_mismatch_penalty = label_mismatch_penalty
    self.tp_comp = nn_comparators.OTMANNDistanceComputer(mlp_layer_labels,
                     label_mismatch_penalty,
                     self.non_assignment_penalty,
                     nn_comparators.MLP_STRUCTURAL_PENALTY_GROUPS,
                     nn_comparators.PATH_LENGTH_TYPES
                     )
    self.mislabel_coeffs = [2.0, 2.0, 1.0, 1.0, 1.0]
    self.struct_coeffs = [0.25, 0.5, 1.0, 2.0, 4.0]
    self.lp_betas = [1e-6] * len(self.struct_coeffs)
    self.emd_betas = [1] * len(self.struct_coeffs)
    self.mlps = generate_mlp_architectures()

  def test_instantiation_and_computation(self):
    """ Testing instantiation. """
    self.report('Testing instantiation of DistSumNNKernelTestCase and computation ' +
                'for specific networks.')
    dist_type_vals = ['lp', 'emd', 'lp-emd']
    all_kernels = []
    for dist_type in dist_type_vals:
      if dist_type == 'lp':
        betas = self.lp_betas
        scales = [1]
      elif dist_type == 'emd':
        betas = self.emd_betas
        scales = [1]
      else:
        betas = [j for i in zip(self.lp_betas, self.emd_betas) for j in i]
        scales = [1, 1]
      tp_kernel = nn_comparators.generate_otmann_kernel_from_params('sum',
                    self.all_layer_labels, self.label_mismatch_penalty,
                    self.non_assignment_penalty,
                    nn_comparators.MLP_STRUCTURAL_PENALTY_GROUPS,
                    nn_comparators.PATH_LENGTH_TYPES,
                    self.mislabel_coeffs, self.struct_coeffs, betas,
                    scales, dist_type=dist_type)
      nn_K = tp_kernel(self.mlps)
      nn_eig_vals, _ = np.linalg.eig(nn_K)
      self.report('dist-type: %s, eigvals: %s.'%(dist_type,
                     get_list_of_floats_as_str(sorted(nn_eig_vals))))
      assert nn_K.shape == (len(self.mlps), len(self.mlps))
      self.report('%s transport kernel:\n%s'%(dist_type,
                  str(np.round(nn_K, 3))), 'test_result')
      assert np.all(np.diag(nn_K) == sum(scales))
      all_kernels.append(nn_K)
    # Check if it is in fact the sum
    if 'lp' in dist_type_vals and 'emd' in dist_type_vals and 'lp-emd' in dist_type_vals:
      lp_kernel = all_kernels[dist_type_vals.index('lp')]
      emd_kernel = all_kernels[dist_type_vals.index('emd')]
      lpemd_kernel = all_kernels[dist_type_vals.index('lp-emd')]
      assert np.linalg.norm(lpemd_kernel - lp_kernel - emd_kernel) < _TOL

  def test_sum_product_equivalence(self):
    """ Unit-test for testing that both kernels compute the same thing in certain cases.
    """
    dist_type_vals = ['lp', 'emd']
    for dist_type in dist_type_vals:
      if dist_type == 'lp':
        betas = self.lp_betas
        scales = [1]
      elif dist_type == 'emd':
        betas = self.emd_betas
        scales = [1]
      sum_kernel = nn_comparators.generate_otmann_kernel_from_params('sum',
                     self.all_layer_labels, self.label_mismatch_penalty,
                     self.non_assignment_penalty,
                     nn_comparators.MLP_STRUCTURAL_PENALTY_GROUPS,
                     nn_comparators.PATH_LENGTH_TYPES,
                     self.mislabel_coeffs, self.struct_coeffs, betas,
                     scales, dist_type=dist_type)
      prod_kernel = nn_comparators.generate_otmann_kernel_from_params('prod',
                      self.all_layer_labels, self.label_mismatch_penalty,
                      self.non_assignment_penalty,
                      nn_comparators.MLP_STRUCTURAL_PENALTY_GROUPS,
                      nn_comparators.PATH_LENGTH_TYPES,
                      self.mislabel_coeffs, self.struct_coeffs, betas,
                      scales, dist_type=dist_type)
      sum_nn_K = sum_kernel(self.mlps)
      prod_nn_K = prod_kernel(self.mlps)
      assert np.linalg.norm(sum_nn_K - prod_nn_K) < _TOL


if __name__ == '__main__':
  execute_tests()

