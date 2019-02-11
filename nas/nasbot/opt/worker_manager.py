"""
  A manager for multiple workers.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=abstract-class-little-used
# pylint: disable=invalid-name
# pylint: disable=no-member

from argparse import Namespace
from multiprocessing import Process
import numpy as np
import os
import shutil
import time
# Local
from ..opt.function_caller import EVAL_ERROR_CODE

TIME_TOL = 1e-5


class WorkerManager(object):
  """ A Base class for a worker manager. """

  def __init__(self, worker_ids, poll_time):
    """ Constructor. """
    if hasattr(worker_ids, '__iter__'):
      self.worker_ids = worker_ids
    else:
      self.worker_ids = list(range(worker_ids))
    self.num_workers = len(self.worker_ids)
    self.poll_time = poll_time
    # These will be set in reset
    self.optimiser = None
    self.latest_results = None
    # Reset
    self.reset()

  def reset(self):
    """ Resets everything. """
    self.optimiser = None
    self.latest_results = [] # A list of namespaces
    self._child_reset()

  def _child_reset(self):
    """ Child reset. """
    raise NotImplementedError('Implement in a child class.')

  def fetch_latest_results(self):
    """ Returns the latest results. """
    ret_idxs = []
    for i in range(len(self.latest_results)):
      if (self.latest_results[i].receive_time <=
            self.optimiser.get_curr_spent_capital() + TIME_TOL):
        ret_idxs.append(i)
    keep_idxs = [i for i in range(len(self.latest_results)) if i not in ret_idxs]
    ret = [self.latest_results[i] for i in ret_idxs]
    self.latest_results = [self.latest_results[i] for i in keep_idxs]
    return ret

  def close_all_jobs(self):
    """ Closes all jobs. """
    raise NotImplementedError('Implement in a child class.')

  def set_optimiser(self, optimiser):
    """ Set the optimiser. """
    self.optimiser = optimiser

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    raise NotImplementedError('Implement in a child class.')

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    raise NotImplementedError('Implement in a child class.')

  def _dispatch_evaluation(self, func_caller, point, qinfo):
    """ Dispatches job. """
    raise NotImplementedError('Implement in a child class.')

  def dispatch_single_evaluation(self, func_caller, point, qinfo):
    """ Dispatches job. """
    raise NotImplementedError('Implement in a child class.')

  def dispatch_batch_of_evaluations(self, func_caller, points, qinfos):
    """ Dispatches an entire batch of evaluations. """
    raise NotImplementedError('Implement in a child class.')

  def get_time_distro_info(self):
    """ Returns information on the time distribution. """
    #pylint: disable=no-self-use
    return ''


# A synthetic worker manager - for simulating multiple workers ---------------------------
class SyntheticWorkerManager(WorkerManager):
  """ A Worker manager for synthetic functions. Mostly to be used in simulations. """

  def __init__(self, num_workers, time_distro='const', time_distro_params=None):
    """ Constructor. """
    self.worker_pipe = None
    super(SyntheticWorkerManager, self).__init__(num_workers, poll_time=None)
    # Set up the time sampler
    self.time_distro = time_distro
    self.time_distro_params = time_distro_params
    self.time_sampler = None
    self._set_up_time_sampler()

  def _set_up_time_sampler(self):
    """ Set up the sampler for the time random variable. """
    self.time_distro_params = Namespace() if self.time_distro_params is None else \
                              self.time_distro_params
    if self.time_distro == 'const':
      if not hasattr(self.time_distro_params, 'const_val'):
        self.time_distro_params.const_val = 1
      self.time_sampler = lambda num_samples: (np.ones((num_samples,)) *
                                               self.time_distro_params.const_val)
    elif self.time_distro == 'uniform':
      if not hasattr(self.time_distro_params, 'ub'):
        self.time_distro_params.ub = 2.0
        self.time_distro_params.lb = 0.0
      ub = self.time_distro_params.ub
      lb = self.time_distro_params.lb
      self.time_sampler = lambda num_samples: (np.random.random((num_samples,)) *
                                               (ub - lb) + lb)
    elif self.time_distro == 'halfnormal':
      if not hasattr(self.time_distro_params, 'ub'):
        self.time_distro_params.sigma = np.sqrt(np.pi/2)
      self.time_sampler = lambda num_samples: np.abs(np.random.normal(
        scale=self.time_distro_params.sigma, size=(num_samples,)))
    else:
      raise NotImplementedError('Not implemented time_distro = %s yet.'%(
                                self.time_distro))

  def _child_reset(self):
    """ Child reset. """
    self.worker_pipe = [[wid, 0.0] for wid in self.worker_ids]

  def sort_worker_pipe(self):
    """ Sorts worker pipe by finish time. """
    self.worker_pipe.sort(key=lambda x: x[-1])

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    return self.worker_pipe[0][-1] # Always return true as this is synthetic.

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    return self.worker_pipe[-1][-1]

  def close_all_jobs(self):
    """ Close all jobs. """
    pass

  def _dispatch_evaluation(self, func_caller, point, qinfo, worker_id, **kwargs):
    """ Dispatch evaluation. """
    qinfo.worker_id = worker_id # indicate which worker
    val, qinfo = func_caller.eval_single(point, qinfo, **kwargs)
    qinfo.eval_time = float(self.time_sampler(1))
    qinfo.val = val
    qinfo.receive_time = qinfo.send_time + qinfo.eval_time
    # Store the result in latest_results
    self.latest_results.append(qinfo)
    return qinfo

  def dispatch_single_evaluation(self, func_caller, point, qinfo, **kwargs):
    """ Dispatch a single evaluation. """
    worker_id = self.worker_pipe[0][0]
    qinfo = self._dispatch_evaluation(func_caller, point, qinfo, worker_id, **kwargs)
    # Sort the pipe
    self.worker_pipe[0][-1] = qinfo.receive_time
    self.sort_worker_pipe()

  def dispatch_batch_of_evaluations(self, func_caller, points, qinfos, **kwargs):
    """ Dispatches an entire batch of evaluations. """
    assert len(points) == self.num_workers
    for idx in range(self.num_workers):
      qinfo = self._dispatch_evaluation(func_caller, points[idx], qinfos[idx],
                                             self.worker_pipe[idx][0], **kwargs)
      self.worker_pipe[idx][-1] = qinfo.receive_time
    self.sort_worker_pipe()

  def get_time_distro_info(self):
    """ Returns information on the time distribution. """
    return self.time_distro


# Real worker manager - for simulating multiple workers --------------------------------
class RealWorkerManager(WorkerManager):
  """ A worker manager for resnet. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, worker_ids, tmp_dir, poll_time=0.5):
    """ Constructor. """
    super(RealWorkerManager, self).__init__(worker_ids, poll_time)
    self.tmp_dir = tmp_dir
    self._rwm_set_up()
    self._child_reset()

  def _rwm_set_up(self):
    """ Sets things up for the child. """
    # Create the result directories. """
    self.result_dir_names = {wid:'%s/result_%s'%(self.tmp_dir, str(wid)) for wid in
                                                 self.worker_ids}
    # Create the working directories
    self.working_dir_names = {wid:'%s/working_%s/tmp'%(self.tmp_dir, str(wid)) for wid in
                                                       self.worker_ids}
    # Create the last receive times
    self.last_receive_times = {wid:0.0 for wid in self.worker_ids}
    # Create file names
    self._result_file_name = 'result.txt'
    self._num_file_read_attempts = 10

  @classmethod
  def _delete_dirs(cls, list_of_dir_names):
    """ Deletes a list of directories. """
    for dir_name in list_of_dir_names:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)

  @classmethod
  def _delete_and_create_dirs(cls, list_of_dir_names):
    """ Deletes a list of directories and creates new ones. """
    for dir_name in list_of_dir_names:
      if os.path.exists(dir_name):
        shutil.rmtree(dir_name)
      os.makedirs(dir_name)

  def _child_reset(self):
    """ Resets child. """
    # Delete/create the result and working directories.
    if not hasattr(self, 'result_dir_names'): # Just for the super constructor.
      return
    self._delete_and_create_dirs(self.result_dir_names.values())
    self._delete_dirs(self.working_dir_names.values())
    self.free_workers = set(self.worker_ids)
    self.qinfos_in_progress = {wid:None for wid in self.worker_ids}
    self.worker_processes = {wid:None for wid in self.worker_ids}

  def _get_result_file_name_for_worker(self, worker_id):
    """ Computes the result file name for the worker. """
    return os.path.join(self.result_dir_names[worker_id], self._result_file_name)

  def _read_result_from_file(self, result_file_name):
    """ Reads the result from the file name. """
    #pylint: disable=bare-except
    num_attempts = 0
    while num_attempts < self._num_file_read_attempts:
      try:
        file_reader = open(result_file_name, 'r')
        read_in = file_reader.read().strip()
        try:
          # try converting to float. If not successful, it is likely an error string.
          read_in = float(read_in)
        except:
          pass
        file_reader.close()
        result = read_in
        break
      except:
        time.sleep(self.poll_time)
        file_reader.close()
        result = EVAL_ERROR_CODE
    return result

  def _read_result_from_worker_and_update(self, worker_id):
    """ Reads the result from the worker. """
    # Read the file
    result_file_name = self._get_result_file_name_for_worker(worker_id)
    val = self._read_result_from_file(result_file_name)
    # Now update the relevant qinfo and put it to latest_results
    qinfo = self.qinfos_in_progress[worker_id]
    qinfo.val = val
    if not hasattr(qinfo, 'true_val'):
      qinfo.true_val = val
    qinfo.receive_time = self.optimiser.get_curr_spent_capital()
    qinfo.eval_time = qinfo.receive_time - qinfo.send_time
    self.latest_results.append(qinfo)
    # Update receive time
    self.last_receive_times[worker_id] = qinfo.receive_time
    # Delete the file.
    os.remove(result_file_name)
    # Delete content in a working directory.
    shutil.rmtree(self.working_dir_names[worker_id])
    # Add the worker to the list of free workers and clear qinfos in progress.
    self.worker_processes[worker_id].terminate()
    self.worker_processes[worker_id] = None
    self.qinfos_in_progress[worker_id] = None
    self.free_workers.add(worker_id)

  def _worker_is_free(self, worker_id):
    """ Checks if worker with worker_id is free. """
    if worker_id in self.free_workers:
      return True
    worker_result_file_name = self._get_result_file_name_for_worker(worker_id)
    if os.path.exists(worker_result_file_name):
      self._read_result_from_worker_and_update(worker_id)
    else:
      return False

  def _get_last_receive_time(self):
    """ Returns the last time we received a job. """
    all_receive_times = self.last_receive_times.values()
    return max(all_receive_times)

  def a_worker_is_free(self):
    """ Returns true if a worker is free. """
    for wid in self.worker_ids:
      if self._worker_is_free(wid):
        return self._get_last_receive_time()
    return None

  def all_workers_are_free(self):
    """ Returns true if all workers are free. """
    all_are_free = True
    for wid in self.worker_ids:
      all_are_free = self._worker_is_free(wid) and all_are_free
    if all_are_free:
      return self._get_last_receive_time()
    else:
      return None

  def _dispatch_evaluation(self, func_caller, point, qinfo, worker_id, **kwargs):
    """ Dispatches evaluation to worker_id. """
    #pylint: disable=star-args
    if self.qinfos_in_progress[worker_id] is not None:
      err_msg = 'qinfos_in_progress: %s,\nfree_workers: %s.'%(
                   str(self.qinfos_in_progress), str(self.free_workers))
      print(err_msg)
      raise ValueError('Check if worker is free before sending evaluation.')
    # First add all the data to qinfo
    qinfo.worker_id = worker_id
    qinfo.working_dir = self.working_dir_names[worker_id]
    qinfo.result_file = self._get_result_file_name_for_worker(worker_id)
    qinfo.point = point
    # Create the working directory
    os.makedirs(qinfo.working_dir)
    # Dispatch the evaluation in a new process
    target_func = lambda: func_caller.eval_single(point, qinfo, **kwargs)
    self.worker_processes[worker_id] = Process(target=target_func)
    self.worker_processes[worker_id].start()
    time.sleep(3)
    # Add the qinfo to the in progress bar and remove from free_workers
    self.qinfos_in_progress[worker_id] = qinfo
    self.free_workers.discard(worker_id)

  def dispatch_single_evaluation(self, func_caller, point, qinfo, **kwargs):
    """ Dispatches a single evaluation to a free worker. """
    worker_id = self.free_workers.pop()
    self._dispatch_evaluation(func_caller, point, qinfo, worker_id, **kwargs)

  def dispatch_batch_of_evaluations(self, func_caller, points, qinfos, **kwargs):
    """ Dispatches a batch of evaluations. """
    assert len(points) == self.num_workers
    for idx in range(self.num_workers):
      self._dispatch_evaluation(func_caller, points[idx], qinfos[idx],
                                self.worker_ids[idx], **kwargs)

  def close_all_jobs(self):
    """ Closes all jobs. """
    pass

  def get_time_distro_info(self):
    """ Returns information on the time distribution. """
    return 'realtime'

