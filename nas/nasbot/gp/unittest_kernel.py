"""
  Unit tests for kernel.py
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used
# pylint: disable=abstract-class-little-used

import numpy as np

# Local imports
from ..gp import kernel
from ..utils.ancillary_utils import dicts_are_equal
from ..utils.base_test_class import BaseTestClass, execute_tests


class KernelBasicTestCase(BaseTestClass):
  """ Basic tests for Kernel function """
  # pylint: disable=too-many-instance-attributes

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(KernelBasicTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Sets up attributes. """
    # Data for generic tests
    self.num_data_1 = 2
    self.num_data_2 = 3
    self.dim = 3
    self.hyper_param_dict_1 = {"param_1":1, "param_2":2, "param_3":3}
    self.hyper_param_dict_2 = {"param_4":4, "param_5":5, "param_6":6}
    self.hyper_param_dict_3 = {"param_1":1, "param_2":2, "param_3":3, "param_4":4,
                               "param_5":5, "param_6":6}
    # Data for the SE and poly kernels
    self.se_scale = 2
    self.data_1 = np.array([[1, 2], [3, 4.5]])
    self.data_2 = np.array([[1, 2], [3, 4]])
    # Data for the squared exponential kernel.
    self.se_dim_bandwidths = [0.1, 1]
    self.true_se_vals_11 = self.se_scale * np.array([[1, np.exp(-406.25/2)],
                                                     [np.exp(-406.25/2), 1]])
    self.true_se_vals_22 = self.se_scale * np.array([[1, np.exp(-404/2)],
                                                     [np.exp(-404/2), 1]])
    self.true_se_vals_12 = self.se_scale * np.array([[1, np.exp(-404/2)],
                                             [np.exp(-406.25/2), np.exp(-0.25/2)]])
    # Data for the polynomial kernel
    self.poly_order = 3
    self.poly_dim_scalings = [0.5, 2]
    self.poly_scale = 2
    self.true_poly_vals_11 = self.poly_scale * np.array([[17.25, 37.75],
                                                         [37.75, 84.25]])**self.poly_order
    self.true_poly_vals_22 = self.poly_scale * np.array([[17.25, 33.75],
                                                         [33.75, 67.25]])**self.poly_order
    self.true_poly_vals_12 = self.poly_scale * np.array([[17.25, 33.75],
                                                         [37.75, 75.25]])**self.poly_order
    # Data for the unscaled polynomial kernel.
    self.unscaled_dim_scalings = [0.8, 0.5, 2]
    self.true_upoly_vals_11 = np.array([[16.89, 37.39], [37.39, 83.89]])**self.poly_order
    self.true_upoly_vals_22 = np.array([[16.89, 33.39], [33.39, 66.89]])**self.poly_order
    self.true_upoly_vals_12 = np.array([[16.89, 33.39], [37.39, 74.89]])**self.poly_order
    # Data for the combined kernel
    self.com_scale = 4.3
    self.true_comb_11 = self.com_scale * self.true_se_vals_11 * self.true_upoly_vals_11
    self.true_comb_22 = self.com_scale * self.true_se_vals_22 * self.true_upoly_vals_22
    self.true_comb_12 = self.com_scale * self.true_se_vals_12 * self.true_upoly_vals_12

  def test_basics(self):
    """ Tests basic functionality. """
    self.report('Testing basic functionality.')
    kern_1 = kernel.Kernel()
    kern_1.set_hyperparams(param_1=1, param_2=2, param_3=3)
    assert dicts_are_equal(kern_1.hyperparams, self.hyper_param_dict_1)
    kern_1.add_hyperparams(**self.hyper_param_dict_2)
    assert dicts_are_equal(kern_1.hyperparams, self.hyper_param_dict_3)
    kern_1.set_hyperparams(**self.hyper_param_dict_2)
    assert dicts_are_equal(kern_1.hyperparams, self.hyper_param_dict_2)

  def test_se_kernel(self):
    """ Tests for the SE kernel. """
    self.report('Tests for the SE kernel.')
    kern = kernel.SEKernel(self.data_1.shape[1], self.se_scale, self.se_dim_bandwidths)
    K11 = kern(self.data_1)
    K22 = kern(self.data_2)
    K12 = kern(self.data_1, self.data_2)
    assert np.linalg.norm(self.true_se_vals_11 - K11) < 1e-10
    assert np.linalg.norm(self.true_se_vals_22 - K22) < 1e-10
    assert np.linalg.norm(self.true_se_vals_12 - K12) < 1e-10

  def test_poly_kernel(self):
    """ Tests for the polynomial kernel. """
    self.report('Tests for the Polynomial kernel.')
    kern = kernel.PolyKernel(self.data_1.shape[1], self.poly_order, self.poly_scale,
                             self.poly_dim_scalings)
    K11 = kern(self.data_1)
    K22 = kern(self.data_2)
    K12 = kern(self.data_1, self.data_2)
    assert np.linalg.norm(self.true_poly_vals_11 - K11) < 1e-10
    assert np.linalg.norm(self.true_poly_vals_22 - K22) < 1e-10
    assert np.linalg.norm(self.true_poly_vals_12 - K12) < 1e-10

  def test_unscaled_poly_kernel(self):
    """ Tests for the polynomial kernel. """
    self.report('Tests for the Unscaled polynomial kernel.')
    kern = kernel.UnscaledPolyKernel(self.data_1.shape[1], self.poly_order,
                                     self.unscaled_dim_scalings)
    K11 = kern(self.data_1)
    K22 = kern(self.data_2)
    K12 = kern(self.data_1, self.data_2)
    assert np.linalg.norm(self.true_upoly_vals_11 - K11) < 1e-10
    assert np.linalg.norm(self.true_upoly_vals_22 - K22) < 1e-10
    assert np.linalg.norm(self.true_upoly_vals_12 - K12) < 1e-10

  def test_comb_kernel(self):
    """ Tests for the combined kernel. """
    self.report('Tests for the Combined kernel.')
    kern_se = kernel.SEKernel(self.data_1.shape[1], self.se_scale, self.se_dim_bandwidths)
    kern_upo = kernel.UnscaledPolyKernel(self.data_1.shape[1], self.poly_order,
                                         self.unscaled_dim_scalings)
    kern = kernel.CoordinateProductKernel(self.data_1.shape[0], self.com_scale,
                                          [kern_se, kern_upo], [[0, 1], [0, 1]])
    K11 = kern(self.data_1)
    K22 = kern(self.data_2)
    K12 = kern(self.data_1, self.data_2)
    assert np.linalg.norm(self.true_comb_11 - K11) < 1e-10
    assert np.linalg.norm(self.true_comb_22 - K22) < 1e-10
    assert np.linalg.norm(self.true_comb_12 - K12) < 1e-10

  def test_effective_length_se(self):
    """ Tests for the effective length in the SE kernel. """
    # pylint: disable=too-many-locals
    self.report('Tests for the effective length in the SE kernel.')
    data_1 = np.array([1, 2])
    data_2 = np.array([[0, 1, 2], [1, 1, 0.5]])
    bws_1 = [0.1, 1]
    bws_2 = [0.5, 1, 2]
    res_l2_1 = np.sqrt(104)
    res_l2_2 = np.array([np.sqrt(2), np.sqrt(5.0625)])
    res_l1_1 = 12
    res_l1_2 = np.array([2, 3.25])
    dim_1 = 2
    dim_2 = 3
    all_data = [(data_1, bws_1, dim_1, res_l2_1, res_l1_1),
                (data_2, bws_2, dim_2, res_l2_2, res_l1_2)]
    for data in all_data:
      kern = kernel.SEKernel(data[2], 1, data[1])
      eff_l1_norms = kern.get_effective_norm(data[0], order=1,
                                             is_single=len(data[0].shape) == 1)
      eff_l2_norms = kern.get_effective_norm(data[0], order=2,
                                             is_single=len(data[0].shape) == 1)
      assert np.linalg.norm(eff_l2_norms - data[3]) < 1e-5
      assert np.linalg.norm(eff_l1_norms - data[4]) < 1e-5

  @classmethod
  def _compute_post_covar(cls, kern, X_tr, X_te):
    """ Computes the posterior covariance. """
    K_tr = kern.evaluate(X_tr, X_tr)
    K_tetr = kern.evaluate(X_te, X_tr)
    K_te = kern.evaluate(X_te, X_te)
    post_covar = K_te - K_tetr.dot(np.linalg.solve(K_tr, K_tetr.T))
    post_std = np.sqrt(np.diag(post_covar))
    return post_covar, post_std

  def test_compute_std_slack_se(self):
    """ Tests for the effective length in the SE kernel. """
    self.report('Tests for std slack in the SE kernel.')
    # The data here are in the order [dim, scale, num_data]
    prob_params = [[2, 1, 10], [3, 2, 0], [10, 6, 13]]
    n = 5
    for prob in prob_params:
      dim_bws = list(np.random.random(prob[0]) * 0.3 + 0.5)
      kern = kernel.SEKernel(prob[0], prob[1], dim_bws)
      X_1 = np.random.random((n, prob[0]))
      X_2 = np.random.random((n, prob[0]))
      X_tr = np.random.random((prob[2], prob[0]))
      _, std_1 = self._compute_post_covar(kern, X_tr, X_1)
      _, std_2 = self._compute_post_covar(kern, X_tr, X_2)
      std_diff = np.abs(std_1 - std_2)
      std_slack = kern.compute_std_slack(X_1, X_2)
      diff_12_scaled = kern.get_effective_norm(X_1 - X_2, order=2, is_single=False)
      kern_diff_12_scaled = kern.hyperparams['scale'] * diff_12_scaled
      assert np.all(std_diff <= std_slack)
      assert np.all(std_slack <= kern_diff_12_scaled)


if __name__ == '__main__':
  execute_tests()

