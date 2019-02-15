"""
  Unit tests for GP-Bandits
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=maybe-no-member
# pylint: disable=invalid-name

import numpy as np
# Local
from nas.nasbot.utils.syn_functions import get_syn_function_caller_from_name
from nas.nasbot.opt.worker_manager import SyntheticWorkerManager
from nas.nasbot.opt import gp_bandit
from nas.nasbot.gp.gp_instances import all_simple_gp_args
# from nn.nn_gp import nn_gp_args
from nas.nasbot.utils.base_test_class import BaseTestClass, execute_tests
from nas.nasbot.utils import reporters
from nas.nasbot.utils.option_handler import load_options


class GPBanditTestCase(BaseTestClass):
  """ Unit tests for gp_bandit.GPBandit. """

  def setUp(self):
    """ Set up. """
    self.func_caller = get_syn_function_caller_from_name('Hartmann6')
    self.worker_manager_1 = SyntheticWorkerManager(1)
    self.worker_manager_4 = SyntheticWorkerManager(4, time_distro='halfnormal')

  def test_instantiation(self):
    """ Tests creation of object. """
    self.report('Testing GP Bandit instantiation.')
    gpb_1 = gp_bandit.GPBandit(self.func_caller, self.worker_manager_1,
                               reporter=reporters.SilentReporter())
    gpb_4 = gp_bandit.GPBandit(self.func_caller, self.worker_manager_4,
                               reporter=reporters.SilentReporter())
    assert gpb_1.domain.get_dim() == self.func_caller.domain.get_dim()
    assert gpb_4.domain.get_dim() == self.func_caller.domain.get_dim()
    self.report('Instantiated GPBandit object.')
    for attr in dir(gpb_4):
      if not attr.startswith('_'):
        self.report('gpb.%s = %s'%(attr, str(getattr(gpb_4, attr))))

  def _test_euc_optimiser_results(self, opt_val, opt_pt, history):
    """ Tests results for a Euclidean GP Bandit optimiser. """
    assert opt_val == history.curr_opt_vals[-1]
    assert opt_pt.shape[0] == self.func_caller.domain.get_dim()

  @classmethod
  def _get_euc_gpb_arguments(cls, num_dims):
    """ Returns options for GP Bandits on a Euclidean space. """
    reporter = reporters.get_reporter('silent')
    gpb_args = gp_bandit.get_all_gp_bandit_args_from_gp_args(all_simple_gp_args)
    options = load_options(gpb_args, reporter=reporter)
    options.get_initial_points = lambda n: np.random.random((n, num_dims))
    options.num_init_evals = 20
    return options

  @classmethod
  def _get_rand_max(cls, func_caller, num_evals):
    """ Gets maximum via random maximising. """
    rand_pts = np.random.random((num_evals, func_caller.dim))
    rand_vals, _ = func_caller.eval_multiple(rand_pts)
    return rand_vals.max()

  def _wrap_up_opt_test(self, test_descr, opt_val, opt_pt, history):
    """ Wrap up for the three optimisation tests below. """
    self._test_euc_optimiser_results(opt_val, opt_pt, history)
    num_evals = len(history.curr_opt_vals)
    best_random = self._get_rand_max(self.func_caller, num_evals)
    self.report('True opt: %0.4f, Found opt: %0.4f, Random: %0.4f, (%d %s evals).'%(
                self.func_caller.opt_val, opt_val, best_random, num_evals,
                test_descr), 'test_result')
    self.report('')

  def test_optimisation_single(self):
    """ Test optimisation of a single point. """
    self.report('Testing GPB on Euclidean space with single worker.')
    options = self._get_euc_gpb_arguments(self.func_caller.domain.get_dim())
    opt_val, opt_pt, history = gp_bandit.gpb_from_func_caller(self.func_caller,
      self.worker_manager_1, 20, mode='asy', acq='ucb', options=options)
    self._wrap_up_opt_test('seq', opt_val, opt_pt, history)

  def test_optimisation_asy(self):
    """ Tests asynchronous optimisation. """
    self.report('Testing GPB on Euclidean space with 4 asynchronous workers.')
    options = self._get_euc_gpb_arguments(self.func_caller.domain.get_dim())
    opt_val, opt_pt, history = gp_bandit.gpb_from_func_caller(self.func_caller,
      self.worker_manager_4, 20, mode='asy', acq='hei', options=options)
    self._wrap_up_opt_test('asy', opt_val, opt_pt, history)

  def test_optimisation_syn(self):
    """ Tests asynchronous optimisation. """
    self.report('Testing GPB on Euclidean space with 4 synchronous workers.')
    options = self._get_euc_gpb_arguments(self.func_caller.domain.get_dim())
    opt_val, opt_pt, history = gp_bandit.gpb_from_func_caller(self.func_caller,
      self.worker_manager_4, 20, mode='syn', acq='hucb', options=options)
    self._wrap_up_opt_test('syn', opt_val, opt_pt, history)


if __name__ == '__main__':
  execute_tests()

