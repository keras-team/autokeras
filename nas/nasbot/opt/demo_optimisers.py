"""
  A test ground for the different optimisers we are implementing.
  -- kandasamy@cs.cmu.edu
"""

#pylint: disable=relative-import

# Local imports
from nas.nasbot.opt import ga_optimiser
from nas.nasbot.nn.syn_nn_functions import cnn_syn_func1, mlp_syn_func1
from nas.nasbot.opt.domains import NNDomain
from nas.nasbot.opt.function_caller import FunctionCaller
from nas.nasbot.nn.nn_constraint_checkers import CNNConstraintChecker, MLPConstraintChecker
from nas.nasbot.nn.nn_modifiers import get_nn_modifier_from_args
from nas.nasbot.opt.nn_opt_utils import get_initial_cnn_pool, get_initial_mlp_pool
from nas.nasbot.utils.reporters import get_reporter
from nas.nasbot.utils.option_handler import load_options
from nas.nasbot.opt.worker_manager import SyntheticWorkerManager


# The problem
NN_TYPE = 'cnn'
CLASS_OR_REG = 'reg'

# Things we won't change much
CAPITAL = 100
NUM_WORKERS = 4
METHODS = ['GA', 'randGA']


def get_opt_problem(nn_type):
  """ Gets parameters for the optimisation problem. """
  if nn_type == 'cnn':
    constraint_checker = CNNConstraintChecker(50, 1e8, 5, 5, 200, 1024, 8)
    init_points = get_initial_cnn_pool()
    func_caller = FunctionCaller(cnn_syn_func1, NNDomain(None, None))
  elif nn_type.startswith('mlp'):
    constraint_checker = MLPConstraintChecker(50, 1e8, 5, 5, 200, 1024, 8)
    init_points = get_initial_mlp_pool(CLASS_OR_REG)
    func_caller = FunctionCaller(mlp_syn_func1, NNDomain(None, None))
  else:
    raise ValueError('Unknown nn_type: %s.'%(nn_type))
  # Common stuff
  mutation_op = get_nn_modifier_from_args(constraint_checker,
                                          [0.5, 0.25, 0.125, 0.075, 0.05])
  init_vals = [func_caller.eval_single(nn)[0] for nn in init_points]
  return constraint_checker, func_caller, mutation_op, init_points, init_vals

def get_options_and_reporter(method, init_points, init_vals):
  """ Returns the options and reporter. """
  reporter = get_reporter('default')
  if method in ['GA', 'randGA']:
    options = load_options(ga_optimiser.ga_opt_args, reporter=reporter)
  else:
    raise ValueError('Unknown method %s.'%(method))
  options.pre_eval_points = init_points
  options.pre_eval_vals = init_vals
  options.pre_eval_true_vals = init_vals
  return options, reporter


def main():
  """ Main function. """
  # Get problem parameters
  worker_manager = SyntheticWorkerManager(NUM_WORKERS, time_distro='halfnormal')
  _, func_caller, mutation_op, init_points, init_vals = get_opt_problem(NN_TYPE)
  print('Best init value: %0.5f' % max(init_vals))

  for method in METHODS:
    print('Method: %s ==========================================================' % method)
    worker_manager.reset()
    # Iterate through each method
    if method in ['GA', 'randGA']:
      is_rand = method == 'randGA'
      options, reporter = get_options_and_reporter(method, init_points, init_vals)
      ga_optimiser.ga_optimise_from_args(func_caller, worker_manager,
        CAPITAL, 'asy', mutation_op, is_rand=is_rand, options=options, reporter=reporter)
    # Wrap up method
    print('')


if __name__ == '__main__':
  main()

