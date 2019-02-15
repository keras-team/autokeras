"""
  Unit tests for NNRandomBandit in nasbot.py.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

# Local
from nas.nasbot.nn.syn_nn_functions import cnn_syn_func1, mlp_syn_func1
from nas.nasbot.opt.function_caller import FunctionCaller
from nas.nasbot.opt import nasbot
from nas.nasbot.opt.unittest_ga_optimiser import test_if_pre_eval_networks_have_changed, \
                                  get_optimiser_args, get_nn_opt_arguments
from nas.nasbot.utils.base_test_class import BaseTestClass, execute_tests
from nas.nasbot.opt.worker_manager import SyntheticWorkerManager


class NNRandomBanditTestCase(BaseTestClass):
  """ Unit test for GA optimisation. """

  def setUp(self):
    """ Set up. """
    ret = get_nn_opt_arguments()
    for key, val in ret.__dict__.iteritems():
      setattr(self, key, val)

  def test_instantiation(self):
    """ Test Creation of NNGPBandit object. """
    self.report('Testing Random Optimiser instantiation.')
    func_caller = FunctionCaller(cnn_syn_func1, self.cnn_domain)
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    optimiser = nasbot.NNRandomBandit(func_caller,
                                            worker_manager, reporter='silent')
    for attr in dir(optimiser):
      if not attr.startswith('_'):
        self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                    'test_result')

  def _get_optimiser_args(self, nn_type):
    """ Returns the options and reporter. """
    return get_optimiser_args(self, nn_type, nasbot.all_nn_random_bandit_args)

  @classmethod
  def _test_optimiser_results(cls, opt_val, _, history, options, options_clone):
    """ Tests optimiser results. """
    assert opt_val == history.curr_opt_vals[-1]
    test_if_pre_eval_networks_have_changed(options, options_clone)

  def test_rand_optimisation_single(self):
    """ Tests optimisation of a single point. """
    self.report('Testing NNRandomBandit with a single worker.')
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    func_caller = FunctionCaller(cnn_syn_func1, self.cnn_domain)
    options, options_clone, reporter, _, _ = self._get_optimiser_args('cnn')
    opt_val, opt_pt, history = nasbot.nnrandbandit_from_func_caller(
      func_caller, worker_manager, 10, 'asy',
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')

  def test_rand_optimisation_asynchronous(self):
    """ Tests optimisation of a single point. """
    self.report('Testing NNRandomBandit with 4 asynchronous workers.')
    worker_manager = SyntheticWorkerManager(4, time_distro='halfnormal')
    func_caller = FunctionCaller(mlp_syn_func1, self.mlp_domain)
    options, options_clone, reporter, _, _ = self._get_optimiser_args('mlp-reg')
    opt_val, opt_pt, history = nasbot.nnrandbandit_from_func_caller(
      func_caller, worker_manager, 5, 'asy',
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')

  def test_rand_optimisation_synchronous(self):
    """ Tests optimisation of a single point. """
    self.report('Testing NNRandomBandit with 4 synchronous workers.')
    worker_manager = SyntheticWorkerManager(4, time_distro='halfnormal')
    func_caller = FunctionCaller(cnn_syn_func1, self.cnn_domain)
    options, options_clone, reporter, _, _ = self._get_optimiser_args('cnn')
    opt_val, opt_pt, history = nasbot.nnrandbandit_from_func_caller(
      func_caller, worker_manager, 5, 'syn',
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')


if __name__ == '__main__':
  execute_tests()

