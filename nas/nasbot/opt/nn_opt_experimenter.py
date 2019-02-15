"""
  Harness for conducting black box optimisation experiments.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=abstract-class-not-used
# pylint: disable=star-args
# pylint: disable=maybe-no-member

from argparse import Namespace
from datetime import datetime
import numpy as np
import os
# Local imports
from nas.nasbot.opt.blackbox_optimiser import Initialiser
from nas.nasbot.opt.ga_optimiser import ga_optimise_from_args
from nas.nasbot.opt.gp_bandit import gpb_from_func_caller
from nas.nasbot.opt.nasbot import nnrandbandit_from_func_caller
from nas.nasbot.opt.nn_opt_utils import get_initial_pool
from nas.nasbot.utils.experimenters import BasicExperimenter

class NNOptExperimenter(BasicExperimenter):
  """ Base class for running experiments. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, experiment_name, func_caller, worker_manager, max_capital, methods,
               num_experiments, save_dir, experiment_options, save_file_prefix='',
               method_options=None, reporter=None, **kwargs):
    """ Constructor. Also see BasicExperimenter for more args. """
    # pylint: disable=too-many-arguments
    save_file_name = self._get_save_file_name(save_dir, experiment_name,
      worker_manager.num_workers, save_file_prefix, worker_manager.get_time_distro_info(),
      max_capital)
    super(NNOptExperimenter, self).__init__(experiment_name, num_experiments,
                                            save_file_name, reporter=reporter, **kwargs)
    self.func_caller = func_caller
    self.worker_manager = worker_manager
    self.max_capital = float(max_capital)
    # Methods
    self.methods = methods
    self.num_methods = len(methods)
    self.domain = func_caller.domain
    self.method_options = (method_options if method_options else
                           {key: None for key in methods})
    # Experiment options will have things such as if the evaluations are noisy,
    # the time distributions etc.
    self.experiment_options = experiment_options
    self._set_up_saving()

  @classmethod
  def _get_save_file_name(cls, save_dir, experiment_name, num_workers, save_file_prefix,
                          time_distro_str, max_capital):
    """ Gets the save file name. """
    save_file_prefix = save_file_prefix if save_file_prefix else experiment_name
    save_file_name = '%s-M%d-%s-c%d-%s.mat'%(save_file_prefix, num_workers,
      time_distro_str, int(max_capital), datetime.now().strftime('%m%d-%H%M%S'))
    save_file_name = os.path.join(save_dir, save_file_name)
    return save_file_name

  def _set_up_saving(self):
    """ Runs some routines to set up saving. """
    # Store methods and the options in to_be_saved
    self.to_be_saved.max_capital = self.max_capital
    self.to_be_saved.num_workers = self.worker_manager.num_workers
    self.to_be_saved.methods = self.methods
    self.to_be_saved.method_options = self.method_options # Some error created here.
    self.to_be_saved.time_distro_str = self.worker_manager.get_time_distro_info()
    # Data about the problem
    self.to_be_saved.true_opt_val = (self.func_caller.opt_val
      if self.func_caller.opt_val is not None else -np.inf)
    self.to_be_saved.true_opt_pt = (self.func_caller.opt_pt
      if self.func_caller.opt_pt is not None else 'not-known')
    self.to_be_saved.domain_type = self.domain.get_type()
#     # For the results
    self.data_to_be_saved = ['query_step_idxs',
                             'query_points',
                             'query_vals',
                             'query_true_vals',
                             'query_send_times',
                             'query_receive_times',
                             'query_eval_times',
                             'curr_opt_vals',
                             'curr_true_opt_vals',
                             'num_jobs_per_worker',
                            ]
    self.data_not_to_be_mat_saved.extend(['method_options', 'query_points'])
    self.data_not_to_be_pickled.extend(['method_options'])
    for data_type in self.data_to_be_saved:
      setattr(self.to_be_saved, data_type, self._get_new_empty_results_array())

  def _get_new_empty_results_array(self):
    """ Returns a new empty arrray to be used for saving results. """
#     return np.empty((self.num_methods, 0), dtype=np.object)
    return np.array([[] for _ in range(self.num_methods)], dtype=np.object)

  def _get_new_iter_results_array(self):
    """ Returns an empty array to be used for saving results of current iteration. """
#     return np.empty((self.num_methods, 1), dtype=np.object)
    return np.array([['-'] for _ in range(self.num_methods)], dtype=np.object)

  def _print_method_header(self, full_method_name):
    """ Prints a header for the current method. """
    experiment_header = '-- Exp %d/%d on %s:: %s with cap %0.4f. ----------------------'%(
      self.experiment_iter, self.num_experiments, self.experiment_name, full_method_name,
      self.max_capital)
    self.reporter.writeln(experiment_header)

  def get_iteration_header(self):
    """ Header for iteration. """
    noisy_str = ('no-noise' if self.func_caller.noise_type == 'none' else
                 'noisy(%0.2f)'%(self.func_caller.noise_scale))
    opt_val_str = ('?' if self.func_caller.opt_val is None
                       else '%0.5f'%(self.func_caller.opt_val))
    ret = '%s (M=%d), td: %s, max=%s, max-capital %0.2f, %s'%(self.experiment_name,
      self.worker_manager.num_workers, self.to_be_saved.time_distro_str, opt_val_str,
      self.max_capital, noisy_str)
    return ret

  def _print_method_result(self, method, comp_opt_val, num_evals):
    """ Prints the result for this method. """
    result_str = 'Method: %s achieved max-val %0.5f in %d evaluations.\n'%(method,
                  comp_opt_val, num_evals)
    self.reporter.writeln(result_str)

  def _get_pre_eval_points_and_vals(self):
    """ Gets Initial points for all methods. """
    if self.experiment_options.pre_eval_points == 'generate':
      init_pool = get_initial_pool(self.domain.get_type())
    else:
      # Load from the file.
      raise NotImplementedError('Not written reading results from file yet.')
    # Create an initialiser
    initialiser = Initialiser(self.func_caller, self.worker_manager)
    initialiser.options.get_initial_points = lambda _: init_pool
    initialiser.options.max_num_steps = 0
    _, _, init_hist = initialiser.initialise()
    pre_eval_points = init_hist.query_points
    pre_eval_vals = init_hist.query_vals
    pre_eval_true_vals = init_hist.query_true_vals
    return pre_eval_points, pre_eval_vals, pre_eval_true_vals

  def run_experiment_iteration(self):
    """ Runs each method in self.methods once and stores the results to be saved. """
    curr_iter_results = Namespace()
    for data_type in self.data_to_be_saved:
      setattr(curr_iter_results, data_type, self._get_new_iter_results_array())

    # Fetch pre-evaluation points.
    self.worker_manager.reset()
    (pre_eval_points_for_all_meths, pre_eval_vals_for_all_meths,
     pre_eval_true_vals_for_all_meths) = self._get_pre_eval_points_and_vals()
    self.reporter.writeln('Using %d pre-eval points with values: eval: %s, true: %s'%(
      len(pre_eval_vals_for_all_meths), pre_eval_vals_for_all_meths,
      pre_eval_true_vals_for_all_meths))

    # Will go through each method in this loop.
    for meth_iter in range(self.num_methods):
      curr_method = self.methods[meth_iter]
      curr_options = self.method_options[curr_method]
      # Set pre_eval points and vals
      curr_options.pre_eval_points = pre_eval_points_for_all_meths
      curr_options.pre_eval_vals = pre_eval_vals_for_all_meths
      curr_options.pre_eval_true_vals = pre_eval_true_vals_for_all_meths
      # Reset worker manager
      self.worker_manager.reset()
      self.reporter.writeln('\nResetting worker manager: worker_manager.optimiser:%s'%(
                            str(self.worker_manager.optimiser)))

      # Call the method here.
      self._print_method_header(curr_method)
      _, _, history = optimise_with_method_on_func_caller(curr_method, self.func_caller,
                        self.worker_manager, self.max_capital, meth_options=curr_options,
                        reporter=self.reporter)

      # Now save results for current method
      for data_type in self.data_to_be_saved:
        data = getattr(history, data_type)
        data_pointer = getattr(curr_iter_results, data_type)
        data_pointer[meth_iter, 0] = data
      # Print out results
      comp_opt_val = history.curr_true_opt_vals[-1]
      num_evals = len(history.curr_true_opt_vals)
      self._print_method_result(curr_method, comp_opt_val, num_evals)
      # Save results of current iteration
      self.update_to_be_saved(curr_iter_results)
      self.save_pickle()
      self.save_results()
      # for meth_iter ends here
    # Save here
    self.update_to_be_saved(curr_iter_results)
    self.save_pickle()
    # No need to explicitly save_results() here - it is done by the parent class.

  def update_to_be_saved(self, curr_iter_results):
    """ Updates the results of the data to be saved with curr_iter_results."""
    for data_type in self.data_to_be_saved:
      data = getattr(curr_iter_results, data_type)
      curr_data_to_be_saved = getattr(self.to_be_saved, data_type)
      if curr_data_to_be_saved.shape[1] == self.experiment_iter:
        updated_data_to_be_saved = curr_data_to_be_saved
        updated_data_to_be_saved[:, -1] = data.ravel()
      elif curr_data_to_be_saved.shape[1] < self.experiment_iter:
        updated_data_to_be_saved = np.concatenate((curr_data_to_be_saved, data), axis=1)
      else:
        raise ValueError('Something wrong with data saving.')
      setattr(self.to_be_saved, data_type, updated_data_to_be_saved)


# --------------------------------------------------------------------------------------
def optimise_with_method_on_func_caller(method, func_caller, worker_manager, max_capital,
                                        meth_options, reporter):
  """ This function does the optimisation. """
  # Mode should be specified in the first three letters
  meth_options.mode = method[0:3]
  specific_method = method[3:]
  if specific_method == 'GA':
    return ga_optimise_from_args(func_caller, worker_manager, max_capital,
                                 meth_options.mode, meth_options.mutation_op,
                                 options=meth_options, reporter=reporter)
  elif specific_method in ['HEI', 'EI', 'HTS', 'TS', 'HUCB', 'UCB']:
    meth_options.acq = specific_method
    return nngpb_from_func_caller(func_caller, worker_manager, meth_options.tp_comp,
                                  max_capital, acq=specific_method,
                                  options=meth_options, reporter=reporter)
  elif specific_method == 'RAND':
    meth_options.acq = specific_method
    return nnrandbandit_from_func_caller(func_caller, worker_manager, max_capital,
                                         options=meth_options, reporter=reporter)
  else:
    raise ValueError('Unknown method: %s'%(method))

