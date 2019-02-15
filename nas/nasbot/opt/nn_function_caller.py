"""
  A function caller to work with MLPs.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=no-member
# pylint: disable=signature-differs

from glob import glob
import shutil
import numpy as np
from time import time, sleep
# Local imports
from nas.nasbot.opt.function_caller import FunctionCaller, EVAL_ERROR_CODE
from nas.nasbot.utils.reporters import get_reporter
import tempfile

_DEBUG_ERROR_PROB = 0.1
# _DEBUG_ERROR_PROB = 0.0


class NNFunctionCaller(FunctionCaller):
  """ Function Caller for NN evaluations. """

  def __init__(self, descr, domain, train_params, debug_mode=False,
               debug_function=None, reporter=None, tmp_dir='/tmp'):
    """ Constructor for train params. """
    super(NNFunctionCaller, self).__init__(None, domain,
                                           opt_pt=None, opt_val=None,
                                           noise_type='none', noise_params=None,
                                           descr=descr)
    if debug_mode:
      if debug_function is None:
        raise ValueError('If in debug mode, debug_function cannot be None.')
      self.debug_function = debug_function
    self.train_params = train_params
    self.debug_mode = debug_mode
    self.reporter = get_reporter(reporter)
    self.root_tmp_dir = tmp_dir

  def eval_single(self, nn, qinfo, noisy=False):
    """ Over-rides eval_single. """
    qinfo.val = self._func_wrapper(nn, qinfo)
    if qinfo.val == EVAL_ERROR_CODE:
      self.reporter.writeln(('Error occurred when evaluating %s. Returning ' +
                             'EVAL_ERROR_CODE: %s.')%(nn, EVAL_ERROR_CODE))
    qinfo.true_val = qinfo.val
    qinfo.point = nn
    return qinfo.val, qinfo

  def _func_wrapper(self, nn, qinfo):
    """ Evaluates the function here - mostly a wrapper to decide between
        the synthetic function vs the real function. """
    # pylint: disable=unused-argument
    # pylint: disable=bare-except
    # pylint: disable=broad-except
    if self.debug_mode:
      ret = self._eval_synthetic_function(nn, qinfo)
    else:
      try:
        self.tmp_dir = tempfile.mkdtemp(dir=self.root_tmp_dir)
        ret = self._eval_validation_score(nn, qinfo)
      except Exception as exc:
        self.reporter.writeln('Exception when evaluating %s: %s'%(nn, exc))
        ret = EVAL_ERROR_CODE
    # Write to the file and return
    qinfo.val = ret
    qinfo.true_val = qinfo.val
    qinfo.point = nn
    self._write_result_to_file(ret, qinfo.result_file)
    try:
      shutil.rmtree(self.tmp_dir)
    except:
      pass
    return ret

  def _eval_synthetic_function(self, nn, qinfo):
    """ Evaluates the synthetic function. """
    result = self.debug_function(nn)
    np.random.seed(int(time() * 10 * int(qinfo.worker_id + 1)) % 100000)
#     sleep_time = 10 + 30 * np.random.random()
    sleep_time = 2 + 10 * np.random.random()
#     sleep_time = .02 + 0.1 * np.random.random()
    sleep(sleep_time)
    if np.random.random() < _DEBUG_ERROR_PROB:
      # For debugging, return an error code with small probability
      return EVAL_ERROR_CODE
    else:
      return result

  def _eval_validation_score(self, qinfo, nn):
    """ Evaluates the validation score. """
    # Design your API here. You can use self.training_params to store anything
    # additional you need.
    raise NotImplementedError('Implement this for specific application.')

  @classmethod
  def _write_result_to_file(cls, result, file_name):
    """ Writes the result to the file name. """
    file_handle = open(file_name, 'w')
    file_handle.write(str(result))
    file_handle.close()

