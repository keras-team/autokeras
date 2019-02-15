"""
  A demo of NASBOT on a MLP (Multi-Layer Perceptron) architecture search problem.
  -- kandasamy@cs.cmu.edu
"""

from argparse import Namespace
import numpy as np
import time
import os
# Local
from nas.nasbot.nn.nn_constraint_checkers import get_nn_domain_from_constraints
from nas.nasbot.nn.nn_visualise import visualise_nn
from nas.nasbot.opt import nasbot
from nas.nasbot.demos.mlp_function_caller import MLPFunctionCaller
from nas.nasbot.opt.worker_manager import RealWorkerManager
from nas.nasbot.utils.reporters import get_reporter

# Data
# The data should be in a pickle file stored as a dictionary. The 'train' key
# should point to the training data while 'vali' points to the validation data.
# In both cases
# For example, after data = pic.load(file_name), data['train']['x'] should point
# to the features of the training data.
# The slice and indoor_location datasets are available at
# http://www.cs.cmu.edu/~kkandasa/nasbot_datasets.html as examples. Put them in the demos
# directory to run the demo.

# Results
# The progress of optimization will be logged in mlp_experiment_dir_<time>/log where
# <time> is a time stamp.

# DATASET = 'slice'
DATASET = 'indoor'

# Search space
MAX_NUM_LAYERS = 60 # The maximum number of layers
MIN_NUM_LAYERS = 5 # The minimum number of layers
MAX_MASS = np.inf # Mass is the total amount of computation at all layers
MIN_MASS = 0
MAX_IN_DEGREE = 5 # Maximum in degree of each layer
MAX_OUT_DEGREE = 55 # Maximum out degree of each layer
MAX_NUM_EDGES = 200 # Maximum number of edges in the network
MAX_NUM_UNITS_PER_LAYER = 1024 # Maximum number of computational units ...
MIN_NUM_UNITS_PER_LAYER = 8    # ... (neurons/conv-filters) per layer.

# Which GPU IDs are available
# GPU_IDS = [0, 1]
GPU_IDS = [0, 3]

# Where to store temporary model checkpoints
EXP_DIR = 'mlp_experiment_dir_%s'%(time.strftime('%Y%m%d%H%M%S'))
LOG_FILE = os.path.join(EXP_DIR, 'log')
TMP_DIR = '/tmp'
os.mkdir(EXP_DIR)

# Function to return the name of the file containing dataset
def get_train_file_name(dataset):
  """ Return train params. """
  # get file name
  if dataset == 'slice':
    train_pickle_file = 'SliceLocalization.p'
  elif dataset == 'indoor':
    train_pickle_file = 'IndoorLoc.p'
  return train_pickle_file

# Specify the budget (in seconds)
BUDGET = 2 * 24 * 60 * 60

# Obtain a reporter object
# REPORTER = get_reporter('default') # Writes results to stdout
REPORTER = get_reporter(open(LOG_FILE, 'w')) # Writes to file log_mlp

def main():
  """ Main function. """
  # Obtain the search space
  nn_domain = get_nn_domain_from_constraints('mlp-reg', MAX_NUM_LAYERS, MIN_NUM_LAYERS,
                MAX_MASS, MIN_MASS, MAX_IN_DEGREE, MAX_OUT_DEGREE, MAX_NUM_EDGES,
                MAX_NUM_UNITS_PER_LAYER, MIN_NUM_UNITS_PER_LAYER)
  # Obtain a worker manager: A worker manager (defined in opt/worker_manager.py) is used
  # to manage (possibly) multiple workers. For a RealWorkerManager, the budget should be
  # given in wall clock seconds.
  worker_manager = RealWorkerManager(GPU_IDS, EXP_DIR)
  # Obtain a function caller: A function_caller is used to evaluate a function defined on
  # neural network architectures. We have defined the MLPFunctionCaller in
  # demos/mlp_function_caller.py. The train_params can be used to specify additional
  # training parameters such as the learning rate etc.
  train_params = Namespace(data_train_file=get_train_file_name(DATASET))
  func_caller = MLPFunctionCaller(DATASET, nn_domain, train_params,
                                  reporter=REPORTER, tmp_dir=TMP_DIR)

  # Run nasbot
  opt_val, opt_nn, _ = nasbot.nasbot(func_caller, worker_manager, BUDGET,
                                     reporter=REPORTER)

  # Print the optimal value and visualise the best network.
  REPORTER.writeln('\nOptimum value found: %0.5f'%(opt_val))
  visualise_file = os.path.join(EXP_DIR, 'mlp_optimal_network')
  REPORTER.writeln('Optimal network visualised in %s.eps.'%(visualise_file))
  visualise_nn(opt_nn, visualise_file)

  # N.B: See function nasbot and class NASBOT in opt/nasbot.py to customise additional
  # parameters of the algorithm.


if __name__ == '__main__':
  main()

