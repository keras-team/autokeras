"""
  Unit-tests for nn_opt_utils.py
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=no-member

import os
import numpy as np
# Local
from ..nn import syn_nn_functions
from ..opt.domains import NNDomain
from ..opt import nn_opt_utils
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..nn.nn_examples import generate_many_neural_networks
from ..nn.nn_visualise import visualise_list_of_nns

_TOL = 1e-5


class NNOptUtilsTestClass(BaseTestClass):
  """ Contains unit tests for general functions in nn_opt_utils.py. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNOptUtilsTestClass, self).__init__(*args, **kwargs)
    num_nns = 20
    self.cnns = generate_many_neural_networks('cnn', num_nns)
    self.mlps = generate_many_neural_networks('mlp-reg', num_nns)

  def test_nnfunction_caller(self):
    """ Unit tests for the Neural Network Function caller. """
    self.report('Testing NNFunctionCaller class.')
    test_cases = [(self.cnns, syn_nn_functions.cnn_syn_func1),
                  (self.mlps, syn_nn_functions.mlp_syn_func1)]
    for nns, func in test_cases:
      func_caller = nn_opt_utils.NNFunctionCaller(func, NNDomain(None, None))
      true_vals = np.array([func(nn) for nn in nns])
      single_result = np.array([func_caller.eval_single(nn)[0] for nn in nns])
      mult_result = np.array(func_caller.eval_multiple(nns)[0])
      assert np.linalg.norm(true_vals - single_result) < _TOL * np.linalg.norm(true_vals)
      assert np.linalg.norm(true_vals - mult_result) < _TOL * np.linalg.norm(true_vals)

  def test_initial_pools(self):
    """ Unit test for initial pools. """
    self.report('Testing initial pools for CNNs and MLPs.')
    save_dir = '../scratch/init_pools'
    init_cnns = nn_opt_utils.get_initial_cnn_pool()
    init_mlps = nn_opt_utils.get_initial_mlp_pool('reg')
    visualise_list_of_nns(init_cnns, os.path.join(save_dir, 'cnns'))
    visualise_list_of_nns(init_mlps, os.path.join(save_dir, 'mlps'))


if __name__ == '__main__':
  execute_tests()

