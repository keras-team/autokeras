"""
  A demo of nasbot on a synthetic function.
  -- kandasamy@cs.cmu.edu
"""

import numpy as np
# Local
from nas.nasbot.nn.nn_constraint_checkers import get_nn_domain_from_constraints
from nas.nasbot.nn.syn_nn_functions import cnn_syn_func1
from nas.nasbot.nn.nn_visualise import visualise_nn
from nas.nasbot.opt import nasbot
from nas.nasbot.opt.function_caller import FunctionCaller
from nas.nasbot.opt.worker_manager import SyntheticWorkerManager


# Search space
MAX_NUM_LAYERS = 50 # The maximum number of layers
MIN_NUM_LAYERS = 5 # The minimum number of layers
MAX_MASS = np.inf # Mass is the total amount of computation at all layers
MIN_MASS = 0
MAX_IN_DEGREE = 5 # Maximum in degree of each layer
MAX_OUT_DEGREE = 55 # Maximum out degree of each layer
MAX_NUM_EDGES = 200 # Maximum number of edges in the network
MAX_NUM_UNITS_PER_LAYER = 1024 # Maximum number of computational units ...
MIN_NUM_UNITS_PER_LAYER = 8    # ... (neurons/conv-filters) per layer.

def main():
  """ Main function. """
  # Obtain the search space
  nn_domain = get_nn_domain_from_constraints('cnn', MAX_NUM_LAYERS, MIN_NUM_LAYERS,
                MAX_MASS, MIN_MASS, MAX_IN_DEGREE, MAX_OUT_DEGREE, MAX_NUM_EDGES,
                MAX_NUM_UNITS_PER_LAYER, MIN_NUM_UNITS_PER_LAYER)
  # Obtain a worker manager: A worker manager (defined in opt/worker_manager.py) is used
  # to manage (possibly) multiple workers. For a synthetic experiment, we will use a
  # synthetic worker manager with 1 worker.
  worker_manager = SyntheticWorkerManager(1)
  # Obtain a function caller: A function_caller is used to evaluate a function defined on
  # neural network architectures. Here, we have obtained a function_caller from a
  # synthetic function, but for real experiments, you might have to write your own caller.
  # See the MLP/CNN demos for an example.
  func_caller = FunctionCaller(cnn_syn_func1, nn_domain)
  # Finally, specify the budget. In this case, it will be just the number of evaluations.
  budget = 20

  # Run nasbot
  opt_val, opt_nn, _ = nasbot.nasbot(func_caller, worker_manager, budget)

  # Print the optimal value and visualise the best network.
  print('\nOptimum value found: ', opt_val)
  print('Optimal network visualised in syn_opt_network.eps.')
  visualise_nn(opt_nn, 'syn_opt_network')

  # N.B: See function nasbot and class NASBOT in opt/nasbot.py to customise additional
  # parameters of the algorithm.


if __name__ == '__main__':
  main()

