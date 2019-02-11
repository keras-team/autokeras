"""
  Function caller for the CNN experiments.
  -- willie@cs.cmu.edu
"""

# pylint: disable=arguments-differ

import os
from time import sleep
# Local
from ..opt.nn_function_caller import NNFunctionCaller
from ..cg.cifar import run_tensorflow_cifar
import traceback

_MAX_TRIES = 3
_SLEEP_BETWEEN_TRIES_SECS = 3

def get_default_cnn_tf_params():
  """ Default MLP training parameters for tensorflow. """
  return {
    'trainBatchSize':32,
    'valiBatchSize':32,
    'trainNumStepsPerLoop':4000,
    'valiNumStepsPerLoop':313,
    'numLoops':20,
    'learningRate':0.005
    }

class CNNFunctionCaller(NNFunctionCaller):
  """ Function caller to be used in the MLP experiments. """

  def __init__(self, *args, **kwargs):
    super(CNNFunctionCaller, self).__init__(*args, **kwargs)
    # Load data
    self.data_file_str = self.train_params.data_dir
    # Check tf_params
    if not hasattr(self.train_params, 'tf_params'):
      self.train_params.tf_params = get_default_cnn_tf_params()

  def _eval_validation_score(self, nn, qinfo, noisy=False):
    # pylint: disable=unused-argument
    # pylint: disable=bare-except
    """ Evaluates the validation score. """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(qinfo.worker_id)
    num_tries = 0
    succ_eval = False
#     self.reporter.writeln('Evaluating %s on GPU %d.'%(nn, qinfo.worker_id))
    while num_tries < _MAX_TRIES and not succ_eval:
      try:
        vali_error = run_tensorflow_cifar.compute_validation_error(nn, self.data_file_str,
                      qinfo.worker_id, self.train_params.tf_params, self.tmp_dir)
        succ_eval = True
        sleep(_SLEEP_BETWEEN_TRIES_SECS)
      except:
        num_tries += 1
        self.reporter.writeln('********* Failed on try %d with gpu %d.'%(
                              num_tries, qinfo.worker_id))
        traceback.print_exc()
        traceback.print_exc(file=self.reporter.out)
    return vali_error

