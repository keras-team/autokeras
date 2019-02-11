"""
  Unit tests for GP bandit with NNs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

# Local
from ..nn.syn_nn_functions import cnn_syn_func1, mlp_syn_func1
from ..opt.function_caller import FunctionCaller
from ..opt import nasbot
from ..nn.nn_comparators import get_default_otmann_distance
from ..opt.unittest_ga_optimiser import test_if_pre_eval_networks_have_changed, \
                                  get_optimiser_args, get_nn_opt_arguments
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..opt.worker_manager import SyntheticWorkerManager


def get_tp_comp(nn_type):
  """ Returns a transport distance computer. """
  tp_comp = get_default_otmann_distance(nn_type, 1.0)
  return tp_comp

class NASBOTTestCase(BaseTestClass):
  """ Unit test for GA optimisation. """
  # TODO: This repeats a lot of code from other unit tests for optimisers such as
  # unittest_ga_optimiser and unittest_gp_bandit etc. Maybe write a parent class
  # for all unit tests for optimisers.

  def setUp(self):
    """ Set up. """
    ret = get_nn_opt_arguments()
    for key, val in ret.__dict__.iteritems():
      setattr(self, key, val)

  def test_instantiation(self):
    """ Test Creation of NASBOT object. """
    self.report('Testing Random Optimiser instantiation.')
    tp_comp = get_tp_comp('cnn')
    func_caller = FunctionCaller(cnn_syn_func1, self.cnn_domain)
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    optimiser = nasbot.NASBOT(func_caller, worker_manager, tp_comp, reporter='silent')
    for attr in dir(optimiser):
      if not attr.startswith('_'):
        self.report('optimiser.%s = %s'%(attr, str(getattr(optimiser, attr))),
                    'test_result')

  def _get_optimiser_args(self, nn_type):
    """ Returns the options and reporter. """
    return get_optimiser_args(self, nn_type, nasbot.all_nasbot_args)

  @classmethod
  def _test_optimiser_results(cls, opt_val, _, history, options, options_clone):
    """ Tests optimiser results. """
    assert opt_val == history.curr_opt_vals[-1]
    test_if_pre_eval_networks_have_changed(options, options_clone)

  def test_nasbot_optimisation_single(self):
    """ Tests optimisation with a single worker. """
    self.report('Testing NASBOT with a single worker.')
    worker_manager = SyntheticWorkerManager(1, time_distro='const')
    func_caller = FunctionCaller(cnn_syn_func1, self.cnn_domain)
    tp_comp = get_tp_comp('cnn')
    options, options_clone, reporter, _, _ = self._get_optimiser_args('cnn')
    opt_val, opt_pt, history = nasbot.nasbot(
      func_caller, worker_manager, 10, tp_comp,
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')

  def test_nasbot_optimisation_asynchronous(self):
    """ Tests optimisation of a single point. """
    self.report('Testing NASBOT with 4 asynchronous workers.')
    worker_manager = SyntheticWorkerManager(4, time_distro='halfnormal')
    func_caller = FunctionCaller(mlp_syn_func1, self.mlp_domain)
    tp_comp = get_tp_comp('mlp-reg')
    options, options_clone, reporter, _, _ = self._get_optimiser_args('mlp-reg')
    opt_val, opt_pt, history = nasbot.nasbot(
      func_caller, worker_manager, 5, tp_comp,
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')

  def test_nasbot_optimisation_synchronous(self):
    """ Tests optimisation of a single point. """
    self.report('Testing NASBOT with 4 synchronous workers.')
    worker_manager = SyntheticWorkerManager(4, time_distro='halfnormal')
    func_caller = FunctionCaller(cnn_syn_func1, self.cnn_domain)
    tp_comp = get_tp_comp('cnn')
    options, options_clone, reporter, _, _ = self._get_optimiser_args('cnn')
    opt_val, opt_pt, history = nasbot.nasbot(
      func_caller, worker_manager, 5, tp_comp,
      options=options, reporter=reporter)
    self._test_optimiser_results(opt_val, opt_pt, history, options, options_clone)
    self.report('')


if __name__ == '__main__':
  execute_tests()

