"""
  Unit tests for function caller.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=maybe-no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=abstract-class-not-used

import numpy as np
# Local
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.syn_functions import get_syn_function_caller_from_name


class FunctionCallerTestCase(BaseTestClass):
  """ Unit tests for FunctionCaller class. """

  def __init__(self, *args, **kwargs):
    super(FunctionCallerTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Set up attributes. """
    self.synthetic_functions = ['hartmann3', 'hartmann6', 'hartmann-23', 'shekel',
                                'branin-20', 'branin-31', 'shekel-40']
    self.num_samples = 10000

  def test_max_vals(self):
    """ Tests for the maximum of the function. """
    self.report('Testing for Maximum value. ')
    for test_fn_name in self.synthetic_functions:
      caller = get_syn_function_caller_from_name(test_fn_name)
      self.report('Testing %s with max value %0.4f.'%(test_fn_name, caller.opt_val),
                  'test_result')
      X = np.random.random((self.num_samples, caller.domain.get_dim()))
      eval_vals, _ = caller.eval_multiple(X)
      assert eval_vals.max() <= caller.opt_val
      if caller.opt_pt is not None:
        max_val_norm = caller.eval_single(caller.opt_pt, normalised=True)
        max_val_unnorm = caller.eval_single(caller.raw_opt_pt, normalised=False)
        assert np.abs(max_val_norm[0] - caller.opt_val) < 1e-5
        assert np.abs(max_val_unnorm[0] - caller.opt_val) < 1e-5


if __name__ == '__main__':
  execute_tests()

