"""
  Harness for Blackbox Optimisation. Implements a parent class that can be inherited by
  all methods for black box optimisation.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-little-used

from argparse import Namespace
import time
import numpy as np
# Local imports
from nas.nasbot.opt.function_caller import EVAL_ERROR_CODE
from nas.nasbot.nn.nn_examples import generate_many_neural_networks
from nas.nasbot.utils.option_handler import get_option_specs, load_options
from nas.nasbot.utils.reporters import get_reporter

blackbox_opt_args = [
  get_option_specs('max_num_steps', False, 1e7,
    'If exceeds this many evaluations, stop.'),
  get_option_specs('capital_type', False, 'realtime',
    'Should be one of return_value, cputime, or realtime'),
  get_option_specs('num_init_evals', False, 20,
    ('The number of evaluations for initialisation. If <0, will use default.')),
  get_option_specs('mode', False, 'asy',
    'If \'syn\', uses synchronous parallelisation, else asynchronous.'),
  get_option_specs('build_new_model_every', False, 7,
    'Updates the GP via a suitable procedure every this many iterations.'),
  get_option_specs('report_results_every', False, 1,
    'Report results every this many iterations.'),
  ]

random_opt_args = blackbox_opt_args


class BlackboxOptimiser(object):
  """ BlackboxOptimiser Class. """
  #pylint: disable=attribute-defined-outside-init
  #pylint: disable=too-many-instance-attributes

  # Methods needed for construction -------------------------------------------------
  def __init__(self, func_caller, worker_manager, model=None,
               options=None, reporter=None):
    """ Constructor.
        func_caller is a FunctionCaller instance.
        worker_manager is a WorkerManager instance.
    """
    # Set up attributes.
    self.func_caller = func_caller
    self.domain = func_caller.domain
    self.worker_manager = worker_manager
    self.options = options
    self.reporter = get_reporter(reporter)
    self.model = model
    # Other set up
    self._set_up()

  def _set_up(self):
    """ Some additional set up routines. """
    # Set up some book keeping parameters
    self.available_capital = 0.0
    self.num_completed_evals = 0
    self.step_idx = 0
    # Initialise step idx
    self.curr_opt_val = -np.inf
    self.curr_opt_pt = None
    self.curr_true_opt_val = -np.inf
    self.curr_true_opt_pt = None
    # Initialise worker manager
    self.worker_manager.set_optimiser(self)
    copyable_params_from_worker_manager = ['num_workers']
    for param in copyable_params_from_worker_manager:
      setattr(self, param, getattr(self.worker_manager, param))
    # Other book keeping stuff
    self.last_report_at = 0
    self.last_model_build_at = 0
    self.eval_points_in_progress = []
    self.eval_idxs_in_progress = []
    # Set initial history
    # query infos will maintain a list of namespaces which contain information about
    # the query in send order. Everything else will be saved in receive order.
    self.history = Namespace(query_step_idxs=np.zeros(0),
                             query_points=[],
                             query_vals=[],
                             query_true_vals=[],
                             query_send_times=[],
                             query_receive_times=[],
                             query_eval_times=[],
                             curr_opt_vals=[],
                             curr_true_opt_vals=[],
                             query_infos=[],
                             job_idxs_of_workers={k:[] for k in
                                                  self.worker_manager.worker_ids},
                            )
    # Finally call the child set up.
    self._child_set_up()
    # Post child set up.
    method_prefix = 'asy' if self.is_asynchronous() else 'syn'
    self.full_method_name = method_prefix + '-' + self.method_name
    self.history.full_method_name = self.full_method_name
    # Set pre_eval_points and vals
    self.pre_eval_points = []
    self.pre_eval_vals = []
    self.pre_eval_true_vals = []

  def is_asynchronous(self):
    """ Returns true if asynchronous."""
    return self.options.mode.lower().startswith('asy')

  def _child_set_up(self):
    """ Additional set up for each child. """
    raise NotImplementedError('Implement in a child class.')

  # Book-keeping -------------------------------------------------------------
  def _update_history(self, qinfo):
    """ Data is a namespace which contains a lot of ancillary information.val
        is the function value. """
    # Update the number of jobs done by each worker regardless
    self.history.job_idxs_of_workers[qinfo.worker_id].append(qinfo.step_idx)
    # If what was returned was and error, then ignore
    if qinfo.val == EVAL_ERROR_CODE:
      return
    # First update the optimal point and value.
    if qinfo.val > self.curr_opt_val:
      self.curr_opt_val = qinfo.val
      self.curr_opt_pt = qinfo.point
    if qinfo.true_val > self.curr_true_opt_val:
      self.curr_true_opt_val = qinfo.true_val
      self.curr_true_opt_pt = qinfo.point
    # Now store in history
    self.history.query_step_idxs = np.append(self.history.query_step_idxs,
                                             qinfo.step_idx)
    self.history.query_points.append(qinfo.point)
    self.history.query_vals.append(qinfo.val)
    self.history.query_true_vals.append(qinfo.true_val)
    self.history.query_send_times.append(qinfo.send_time)
    self.history.query_receive_times.append(qinfo.receive_time)
    self.history.query_eval_times.append(qinfo.eval_time)
    self.history.curr_opt_vals.append(self.curr_opt_val)
    self.history.curr_true_opt_vals.append(self.curr_true_opt_val)
    self.history.query_infos.append(qinfo)

  def _get_jobs_for_each_worker(self):
    """ Returns the number of jobs for each worker as a list. """
    jobs_each_worker = [len(elem) for elem in self.history.job_idxs_of_workers.values()]
    if self.num_workers <= 4:
      jobs_each_worker_str = str(jobs_each_worker)
    else:
      return '[min:%d, max:%d]'%(min(jobs_each_worker), max(jobs_each_worker))
    return  jobs_each_worker_str + ' (%d/%d)'%(len(self.history.query_vals),
                                               sum(jobs_each_worker))

  def _get_curr_job_idxs_in_progress(self):
    """ Returns the current job indices in progress. """
    if self.num_workers <= 4:
      return str(self.eval_idxs_in_progress)
    else:
      total_in_progress = len(self.eval_idxs_in_progress)
      min_idx = (-1 if total_in_progress == 0 else min(self.eval_idxs_in_progress))
      max_idx = (-1 if total_in_progress == 0 else max(self.eval_idxs_in_progress))
      dif = -1 if total_in_progress == 0 else max_idx - min_idx
      return '[min:%d, max:%d, dif:%d, tot:%d]'%(min_idx, max_idx, dif, total_in_progress)

  def _report_curr_results(self):
    """ Writes current result to reporter. """
    cap_frac = (np.nan if self.available_capital <= 0 else
                self.get_curr_spent_capital()/self.available_capital)
    report_str = ' '.join(['%s:'%(self.full_method_name),
                           '(%03d)::'%(self.step_idx),
                           'cap: %0.3f,'%(cap_frac),
                           'best_val: (e%0.3f, t%0.3f),'%(
                              self.curr_opt_val, self.curr_true_opt_val),
                           'w=%s,'%(self._get_jobs_for_each_worker()),
                           'inP=%s'%(self._get_curr_job_idxs_in_progress()),
                          ])
    self.reporter.writeln(report_str)
    self.last_report_at = self.step_idx

  # Methods needed for initialisation ----------------------------------------
  def perform_initial_queries(self):
    """ Perform initial queries. """
    # If we already have some pre_eval points then do this.
    if (hasattr(self.options, 'pre_eval_points') and
        self.options.pre_eval_points is not None):
      self.pre_eval_vals = self.options.pre_eval_vals
      self.pre_eval_points = self.options.pre_eval_points
      self.num_pre_eval_points = len(self.options.pre_eval_points)
      self.curr_opt_val = max(self.pre_eval_vals)
      self.curr_opt_pt = self.pre_eval_points[np.argmax(self.pre_eval_points)]
      self.pre_eval_true_vals = self.options.pre_eval_true_vals
      self.curr_true_opt_val = max(self.pre_eval_true_vals)
      self.curr_true_opt_pt = self.pre_eval_points[np.argmax(self.pre_eval_true_vals)]
      return
    # Get the initial points
    num_init_evals = int(self.options.num_init_evals)
    if num_init_evals > 0:
      num_init_evals = max(self.num_workers, num_init_evals)
      init_points = self.options.get_initial_points(num_init_evals)
      self.reporter.write('Initialising with %d points ... '%(len(init_points)))
      for init_step in range(len(init_points)):
        self.step_idx += 1
        self._wait_for_a_free_worker()
        self._dispatch_single_evaluation_to_worker_manager(init_points[init_step])
      self._wait_for_all_free_workers()
      self.reporter.writeln('best initial value = %0.4f.'%(self.curr_opt_val))

  def initialise_capital(self):
    """ Initialises capital. """
    if self.options.capital_type == 'return_value':
      self.spent_capital = 0.0
    elif self.options.capital_type == 'cputime':
      self.init_cpu_time_stamp = time.clock()
    elif self.options.capital_type == 'realtime':
      self.init_real_time_stamp = time.time()

  def get_curr_spent_capital(self):
    """ Computes the current spent time. """
    if self.options.capital_type == 'return_value':
      return self.spent_capital
    elif self.options.capital_type == 'cputime':
      return time.clock() - self.init_cpu_time_stamp
    elif self.options.capital_type == 'realtime':
      return time.time() - self.init_real_time_stamp

  def set_curr_spent_capital(self, spent_capital):
    """ Sets the current spent capital. Useful only in synthetic set ups."""
    if self.options.capital_type == 'return_value':
      self.spent_capital = spent_capital

  def optimise_initialise(self):
    """ Initialisation for optimisation. """
    self.curr_opt_pt = None
    self.curr_opt_val = -np.inf
    self.initialise_capital()
    self.perform_initial_queries()
    self._child_optimise_initialise()

  def _child_optimise_initialise(self):
    """ Any initialisation for a child class. """
    raise NotImplementedError('Implement in a child class.')

  # Methods needed for querying ----------------------------------------------------
  def _wait_till_free(self, is_free, poll_time):
    """ Waits until is_free returns true. """
    keep_looping = True
    while keep_looping:
      last_receive_time = is_free()
      if last_receive_time is not None:
        # Get the latest set of results and dispatch the next job.
        self.set_curr_spent_capital(last_receive_time)
        latest_results = self.worker_manager.fetch_latest_results()
        for qinfo_result in latest_results:
          self._update_history(qinfo_result)
          self._remove_from_in_progress(qinfo_result)
        self._add_data_to_model(latest_results)
        keep_looping = False
      else:
        time.sleep(poll_time)

  def _wait_for_a_free_worker(self):
    """ Checks if a worker is free and updates with the latest results. """
    self._wait_till_free(self.worker_manager.a_worker_is_free,
                         self.worker_manager.poll_time)

  def _wait_for_all_free_workers(self):
    """ Checks to see if all workers are free and updates with latest results. """
    self._wait_till_free(self.worker_manager.all_workers_are_free,
                         self.worker_manager.poll_time)

  def _add_to_in_progress(self, qinfos):
    """ Adds jobs to in progress. """
    for qinfo in qinfos:
      self.eval_idxs_in_progress.append(qinfo.step_idx)
      self.eval_points_in_progress.append(qinfo.point)

  def _remove_from_in_progress(self, qinfo):
    """ Removes a job from the in progress status. """
    completed_eval_idx = self.eval_idxs_in_progress.index(qinfo.step_idx)
    self.eval_idxs_in_progress.pop(completed_eval_idx)
    self.eval_points_in_progress.pop(completed_eval_idx)

  def _dispatch_single_evaluation_to_worker_manager(self, point):
    """ Dispatches an evaluation to the worker manager. """
    # Create a new qinfo namespace and dispatch new job.
    qinfo = Namespace(send_time=self.get_curr_spent_capital(), step_idx=self.step_idx,
                      point=point)
    self.worker_manager.dispatch_single_evaluation(self.func_caller, point, qinfo)
    self._add_to_in_progress([qinfo])

  def _dispatch_batch_of_evaluations_to_worker_manager(self, points):
    """ Dispatches a batch of evaluations to the worker manager. """
    qinfos = [Namespace(send_time=self.get_curr_spent_capital(), step_idx=self.step_idx+j,
                        point=points[j]) for j in range(self.num_workers)]
    self.worker_manager.dispatch_batch_of_evaluations(self.func_caller, points, qinfos)
    self._add_to_in_progress(qinfos)

  # Methods needed for optimisation -------------------------------------------------
  def _terminate_now(self):
    """ Returns true if we should terminate now. """
    if self.step_idx >= self.options.max_num_steps:
      self.reporter.writeln('Exceeded %d evaluations. Terminating Now!'%(
                            self.options.max_num_steps))
      return True
    return self.get_curr_spent_capital() >= self.available_capital

  def add_capital(self, capital):
    """ Adds capital. """
    self.available_capital += float(capital)

  def _determine_next_eval_point(self):
    """ Determine the next point for evaluation. """
    raise NotImplementedError('Implement in a child class.!')

  def _determine_next_batch_of_eval_points(self):
    """ Determine the next batch of eavluation points. """
    raise NotImplementedError('Implement in a child class.!')

  def _post_process_next_eval_point(self, point):
    """ Post-process the next point for evaluation. By default, returns same point. """
    #pylint: disable=no-self-use
    return point

  def _add_data_to_model(self, qinfos):
    """ Adds data to model. """
    qinfos = [qinfo for qinfo in qinfos if qinfo.val != EVAL_ERROR_CODE]
    return self._child_add_data_to_model(qinfos)

  def _child_add_data_to_model(self, qinfos):
    """ Adds data to model in the child class. """
    raise NotImplementedError('Implement in a child class.!')

  def _build_new_model(self):
    """ Builds a new model. """
    self.last_model_build_at = self.step_idx
    self._child_build_new_model()

  def _child_build_new_model(self):
    """ Builds a new model. """
    raise NotImplementedError('Implement in a child class.!')

  def _update_capital(self, qinfos):
    """ Updates the capital according to the cost of the current query. """
    if not hasattr(qinfos, '__iter__'):
      qinfos = [qinfos]
    query_receive_times = []
    for idx in len(qinfos):
      if self.options.capital_type == 'return_value':
        query_receive_times[idx] = qinfos[idx].send_time + qinfos[idx].eval_time
      elif self.options.capital_type == 'cputime':
        query_receive_times[idx] = time.clock() - self.init_cpu_time_stamp
      elif self.options.capital_type == 'realtime':
        query_receive_times[idx] = time.time() - self.init_real_time_stamp
      # Finally add the receive time of the job to qinfo.
      qinfos[idx].receive_time = query_receive_times[idx]
      qinfos[idx].eval_time = qinfos[idx].receive_time - qinfos[idx].send_time
      if qinfos[idx].eval_time < 0:
        raise ValueError(('Something wrong with the timing. send: %0.4f, receive: %0.4f,'
               + ' eval: %0.4f.')%(qinfos[idx].send_time, qinfos[idx].receive_time,
               qinfos[idx].eval_time))
    # Compute the maximum of all receive times
    max_query_receive_times = max(query_receive_times)
    return max_query_receive_times

  # Main optimisation routine ------------
  def _asynchronous_optimise_routine(self):
    """ Optimisation routine for asynchronous part. """
    self._wait_for_a_free_worker()
    next_pt = self._determine_next_eval_point()
    self._dispatch_single_evaluation_to_worker_manager(next_pt)
    self.step_idx += 1

  def _synchronous_optimise_routine(self):
    """ Optimisation routine for the synchronous part. """
    self._wait_for_all_free_workers()
    next_batch_of_points = self._determine_next_batch_of_eval_points()
    self._dispatch_batch_of_evaluations_to_worker_manager(next_batch_of_points)
    self.step_idx += self.num_workers

  def _optimise_wrap_up(self):
    """ Some wrap up before optimisation.
        Particularly store additional data to history. """
    self.worker_manager.close_all_jobs()
    self._wait_for_all_free_workers()
    self._report_curr_results()
    # Store additional data
    self.history.num_jobs_per_worker = np.array(self._get_jobs_for_each_worker())

  def _main_loop_pre(self):
    """ Anything to be done before each iteration of the main loop. Mostly in case
        this is needed by a child class. """
    pass

  def _main_loop_post(self):
    """ Anything to be done after each iteration of the main loop. Mostly in case
        this is needed by a child class. """
    pass

  def optimise(self, max_capital):
    """ This executes the optimisation algorithm. """
    self.add_capital(max_capital)
    self.optimise_initialise()

    # Main Loop --------------------------
    while not self._terminate_now():
      self._main_loop_pre() # Anything to be done before each iteration

      # Optimisation step
      if self.is_asynchronous():
        self._asynchronous_optimise_routine()
      else:
        self._synchronous_optimise_routine()

      # Some Book-keeping
      if self.step_idx - self.last_model_build_at >= self.options.build_new_model_every:
        self._build_new_model()
      if self.step_idx - self.last_report_at >= self.options.report_results_every:
        self._report_curr_results()
      self._main_loop_post() # Anything to be done after each iteration

    # Wrap up and return
    self._optimise_wrap_up()
    return self.curr_opt_val, self.curr_opt_pt, self.history


# Instantiate a random optimiser ==================================================
class RandomOptimiser(BlackboxOptimiser):
  """ A class which optimises just using random evaluations. """
  #pylint: disable=attribute-defined-outside-init

  # Constructor.
  def __init__(self, func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    self.reporter = get_reporter(reporter)
    if options is None:
      options = load_options(blackbox_opt_args, reporter=reporter)
    super(RandomOptimiser, self).__init__(func_caller, worker_manager, model=None,
                                          options=options, reporter=self.reporter)

  def _child_set_up(self):
    """ Additional set up for random optimiser. """
    self.method_name = 'RAND'

  def _child_optimise_initialise(self):
    """ No initialisation for random querying. """
    pass

  def _child_add_data_to_model(self, qinfos):
    """ Update the optimisation model. """
    pass

  def _child_build_new_model(self):
    """ Build new optimisation model. """
    pass

  def _determine_next_eval_point(self):
    """ Determine the next point for evaluation. """
    num_cands = 20
    nns = generate_many_neural_networks('cnn', num_cands)
    rand_idx = np.random.randint(num_cands)
    return nns[rand_idx]

  def _determine_next_batch_of_eval_points(self):
    """ Determine the next point for evaluation. """
    num_cands = max(20, 5 * self.num_workers)
    nns = generate_many_neural_networks('cnn', num_cands)
    rand_idxs = np.random.choice(num_cands, self.num_workers, replace=False)
    return [nns[i] for i in rand_idxs]


# Instantiate an Initialiser ==================================================
# This can be used to evaluate just at a set of initial points ----------------
class Initialiser(BlackboxOptimiser):
  """ A class which used for evaluating at an init pool only. """
  #pylint: disable=attribute-defined-outside-init

  # Constructor.
  def __init__(self, func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    self.reporter = get_reporter(reporter)
    if options is None:
      options = load_options(blackbox_opt_args, reporter=reporter)
    super(Initialiser, self).__init__(func_caller, worker_manager, model=None,
                                          options=options, reporter=self.reporter)

  def _child_set_up(self):
    """ Additional set up for random optimiser. """
    self.method_name = 'Initialiser'

  def initialise(self):
    """ Initialises. """
    return self.optimise(0)

  def _child_optimise_initialise(self):
    """ No initialisation for random querying. """
    pass

  def _child_add_data_to_model(self, qinfos):
    """ Update the optimisation model. """
    pass

  def _child_build_new_model(self):
    """ Build new optimisation model. """
    pass

  def _determine_next_eval_point(self):
    """ Determine the next point for evaluation. """
    # We should not need to override this function.
    pass

  def _determine_next_batch_of_eval_points(self):
    """ Determine the next point for evaluation. """
    # We should not need to override this function.
    pass

# An API for random optimisation ===================================================

# 1. Optimisation from a FunctionCaller object. ------------------------------------
def random_optimise_from_args(func_caller, worker_manager, max_capital,
                              mode=None, options=None, reporter='default'):
  """ Random otpimisation from a utils.function_caller.FunctionCaller instance. """
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(random_opt_args, reporter=reporter)
    options.mode = mode
  return (RandomOptimiser(func_caller, worker_manager, options, reporter)).optimise(
            max_capital)

