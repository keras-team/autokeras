"""
  Unit tests for optimers.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used

from argparse import Namespace
import numpy as np
# Local
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils import oper_utils


class OptimiserTestCase(BaseTestClass):
  """Unit test class for optimisers in oper_utils. """

  def __init__(self, *args, **kwargs):
    super(OptimiserTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Sets up attributes. """
    self.problems = []
    self.max_evals = 2e4
    # First problem
    obj = lambda x: np.dot(x-1, x)
    dim = 4
    min_pt = np.array([0.5] * dim)
    max_pt = np.array([-1] * dim)
    self.problems.append(self._get_test_case_problem_instance(
                         obj, [-1] * dim, [1] * dim, min_pt, max_pt, '4D-quadratic'))
    # Second problem
    obj = lambda x: np.exp(-np.dot(x-0.5, x))
    dim = 2
    min_pt = np.array([-1] * dim)
    max_pt = np.array([0.25] * dim)
    self.problems.append(self._get_test_case_problem_instance(
                         obj, [-1] * dim, [1] * dim, min_pt, max_pt, '2D-gaussian'))

  @classmethod
  def _get_test_case_problem_instance(cls, obj, lb, ub, min_pt, max_pt, descr=''):
    """ A wrapper which returns a problem instance as a list. """
    min_val = float(obj(min_pt))
    max_val = float(obj(max_pt))
    problem_inst = Namespace(obj=obj, dim=len(lb), lb=lb, ub=ub, min_pt=min_pt,
                             max_pt=max_pt, min_val=min_val, max_val=max_val,
                             descr=descr)
    return problem_inst

  def test_random_maximise(self):
    """ Test direct optmisation."""
    self.report('Rand maximise:')
    num_max_successes = 0
    for prob in self.problems:
      prob_bounds = np.concatenate((np.array(prob.lb).reshape(1, -1),
                                    np.array(prob.ub).reshape(1, -1)),
                                    axis=0).T
      max_val_soln, _ = oper_utils.random_maximise(
        prob.obj, prob_bounds, 2 * self.max_evals, False)
      diff = abs(prob.max_val - max_val_soln)
      self.report(prob.descr + '(max):: True: %0.4f, Soln: %0.4f,  diff: %0.4f'%(
                                prob.max_val, max_val_soln, diff), 'test_result')
      max_is_successful = diff < 1e-3
      num_max_successes += max_is_successful
    # Check if successful
    assert num_max_successes >= 1

  def test_direct(self):
    """ Test direct optmisation."""
    self.report('DiRect minimise and maximise:')
    num_min_successes = 0
    num_max_successes = 0
    for prob in self.problems:
      # First the minimimum
      min_val_soln, _, _ = oper_utils.direct_ft_minimise(
        prob.obj, prob.lb, prob.ub, self.max_evals)
      diff = abs(prob.min_val - min_val_soln)
      self.report(prob.descr + '(min):: True: %0.4f, Soln: %0.4f,  diff: %0.4f.'%(
                                prob.min_val, min_val_soln, diff))
      min_is_successful = 1
      num_min_successes += min_is_successful
      # Now the maximum
      max_val_soln, _, _ = oper_utils.direct_ft_maximise(
        prob.obj, prob.lb, prob.ub, self.max_evals)
      diff = abs(prob.max_val - max_val_soln)
      self.report(prob.descr + '(max):: True: %0.4f, Soln: %0.4f,  diff: %0.4f'%(
                                prob.max_val, max_val_soln, diff), 'test_result')
      max_is_successful = 1
      num_max_successes += max_is_successful
    # Check if successful
    assert num_min_successes == len(self.problems)
    assert num_max_successes == len(self.problems)


if __name__ == '__main__':
  execute_tests()

