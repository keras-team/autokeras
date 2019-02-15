"""
  Function caller for the MLP experiments.
  -- willie@cs.cmu.edu
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=arguments-differ

import pickle
import os
from time import sleep
from copy import deepcopy
# Local
from nas.nasbot.opt.nn_function_caller import NNFunctionCaller
from nas.nasbot.cg import run_tensorflow

_MAX_TRIES = 3
_SLEEP_BETWEEN_TRIES_SECS = 3

def get_default_mlp_tf_params():
  """ Default MLP training parameters for tensorflow. """
  return {
    'trainBatchSize':256,
    'valiBatchSize':1000,
    'trainNumStepsPerLoop':100,
    'valiNumStepsPerLoop':5,
    'numLoops':200,
    'learningRate':0.00001,
    }


class MLPFunctionCaller(NNFunctionCaller):
  """ Function caller to be used in the MLP experiments. """

  def __init__(self, *args, **kwargs):
    super(MLPFunctionCaller, self).__init__(*args, **kwargs)
    # Load data
    # with open(self.train_params.data_train_file, 'rb') as input_file:
    data = pickle.load(open(self.train_params.data_train_file, 'rb'), encoding='latin1')
    self.data_train = data['train']
    self.data_vali = data['vali']
    self.reporter.writeln('Loaded data: ' + self.train_params.data_train_file)
    self.reporter.writeln('Training data shape: ' + 'x: ' +
                          str(self.data_train['x'].shape) +
                          ', ' + 'y: ' + str(self.data_train['y'].shape))
    self.reporter.writeln('Validation data shape: ' + 'x: ' +
                          str(self.data_vali['x'].shape) +
                          ', ' + 'y: ' + str(self.data_vali['y'].shape))
    # Check tf_params
    if not hasattr(self.train_params, 'tf_params'):
      self.train_params.tf_params = get_default_mlp_tf_params()
    if self.domain.get_type() == 'mlp-class':
      self.train_params.tf_params = deepcopy(self.train_params.tf_params)
      self.train_params.tf_params['num_classes'] = self.data_train['y'].shape[1]

  def _eval_validation_score(self, nn, qinfo, noisy=False):
    """ Evaluates the validation score. """
    # pylint: disable=bare-except
    # pylint: disable=unused-argument
    os.environ['CUDA_VISIBLE_DEVICES'] = str(qinfo.worker_id)
    num_tries = 0
    succ_eval = False
    while num_tries < _MAX_TRIES and not succ_eval:
      try:
        vali_score = run_tensorflow.compute_validation_error(nn, self.data_train,
                       self.data_vali, 0, self.train_params.tf_params, self.tmp_dir)
        succ_eval = True
      except:
        sleep(_SLEEP_BETWEEN_TRIES_SECS)
        num_tries += 1
        self.reporter.writeln('********* Failed on try %d with gpu %d.'%(
                              num_tries, qinfo.worker_id))
    return vali_score

