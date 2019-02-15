"""
  Harness for GP Bandit Optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class
# pylint: disable=abstract-class-not-used
# pylint: disable=attribute-defined-outside-init

from argparse import Namespace
import numpy as np

# Local imports
from nas.nasbot.opt import gpb_acquisitions
from nas.nasbot.opt.blackbox_optimiser import blackbox_opt_args, BlackboxOptimiser
from nas.nasbot.gp.gp_core import GP
from nas.nasbot.gp.gp_instances import SimpleGPFitter, all_simple_gp_args
from nas.nasbot.utils.option_handler import get_option_specs, load_options
from nas.nasbot.utils.reporters import get_reporter

gp_bandit_args = [
  # Acquisition
  get_option_specs('acq', False, 'ei',
    'Which acquisition to use: ts, ucb, ei, ttei, bucb, ucbpe. If using multiple ' +
    'give them as a hyphen separated list e.g. ucb-ts-ei-ttei'),
  get_option_specs('acq_probs', False, 'uniform',
    'With what probability should we choose each strategy given in acq.'),
  get_option_specs('acq_opt_method', False, 'rand',
    'Which optimiser to use when maximising the acquisition function.'),
  get_option_specs('handle_parallel', False, 'halluc',
    'How to handle parallelisations. Should be halluc or naive.'),
  get_option_specs('acq_opt_max_evals', False, -1,
    'Number of evaluations when maximising acquisition. If negative uses default value.'),
  # The following are perhaps not so important.
  get_option_specs('shrink_kernel_with_time', False, 0,
    'If True, shrinks the kernel with time so that we don\'t get stuck.'),
  get_option_specs('perturb_thresh', False, 1e-4,
    ('If the next point chosen is too close to an exisiting point by this times the '
     'diameter, then we will perturb the point a little bit before querying. This is '
     'mainly to avoid numerical stability issues.')),
  get_option_specs('track_every_time_step', False, 0,
    ('If 1, it tracks every time step.')),
  # The following are for managing GP hyper-parameters. They override hp_tune_criterion
  # and ml_hp_tune_opt from the GP args.
  get_option_specs('gp_hp_tune_criterion', False, 'ml',
                   'Which criterion to use when tuning hyper-parameters. Other ' +
                   'options are post_sampling and post_mean.'),
  get_option_specs('gp_ml_hp_tune_opt', False, 'rand_exp_sampling',
                   'Which optimiser to use when maximising the tuning criterion.'),
  ]

def get_all_gp_bandit_args_from_gp_args(gp_args):
  """ Returns the GP bandit arguments from the arguments for the GP. """
  return gp_args + blackbox_opt_args + gp_bandit_args


# The GPBandit Class
# ========================================================================================
class GPBandit(BlackboxOptimiser):
  """ GPBandit Class. """
  # pylint: disable=attribute-defined-outside-init

  # Constructor.
  def __init__(self, func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    self.gp = None
    if options is None:
      reporter = get_reporter(reporter)
      options = load_options(get_all_gp_bandit_args_from_gp_args(all_simple_gp_args),
                             reporter=reporter)
    super(GPBandit, self).__init__(func_caller, worker_manager, None,
                                   options=options, reporter=reporter)

  def _child_set_up(self):
    """ Some set up for the GPBandit class. """
    # Set up acquisition optimisation
    self._set_up_acq_opt()
    self.method_name = self.options.acq
    self.acqs_to_use = self.options.acq.split('-')
    if self.options.acq_probs == 'uniform':
      self.acq_probs = np.ones(len(self.acqs_to_use)) / float(len(self.acqs_to_use))
    else:
      self.acq_probs = np.array([float(x) for x in self.options.acq_probs.split('-')])
    self.acq_probs = self.acq_probs / self.acq_probs.sum()
    assert len(self.acq_probs) == len(self.acqs_to_use)
    # Override options for hp_tune_criterion and ml_hp_tune_opt
    self.options.hp_tune_criterion = self.options.gp_hp_tune_criterion
    self.options.ml_hp_tune_opt = self.options.gp_ml_hp_tune_opt

  def _get_acq_opt_method(self):
    """ Returns the method for optimising the acquisition. """
    if self.options.acq_opt_method == 'dflt_domain_opt_method':
      return self.domain.dflt_domain_opt_method
    else:
      return self.options.acq_opt_method

  def _set_up_acq_opt(self):
    """ Sets up optimisation for acquisition. """
    # First set up function to get maximum evaluations.
    if isinstance(self.options.acq_opt_max_evals, int):
      if self.options.acq_opt_max_evals > 0:
        self.get_acq_opt_max_evals = lambda t: self.options.acq_opt_max_evals
      else:
        self.get_acq_opt_max_evals = None
    else: # In this case, the user likely passed a function here.
      self.get_acq_opt_max_evals = self.options.acq_opt_max_evals
    # Additional set up based on the specific optimisation procedure
    if self._get_acq_opt_method() == 'direct':
      self._set_up_acq_opt_direct()
    elif self._get_acq_opt_method() == 'rand':
      self._set_up_acq_opt_rand()
    elif self._get_acq_opt_method() == 'ga':
      self._set_up_acq_opt_ga()
    else:
      raise NotImplementedError('Not implemented acquisition optimisation for %s yet.'%(
                                self.options.acq_opt_method))

  def _set_up_acq_opt_direct(self):
    """ Sets up optimisation for acquisition using direct. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 1 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(
        lead_const * np.sqrt(min(t, 1000)), 1000, 3e4)

  def _set_up_acq_opt_rand(self):
    """ Sets up optimisation for acquisition using random search. """
    if self.get_acq_opt_max_evals is None:
      lead_const = 10 * min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(
        lead_const * np.sqrt(min(t, 1000)), 2000, 3e4)

  def _set_up_acq_opt_ga(self):
    """ Sets up optimisation for the acquisition using genetic algorithms. """
    raise NotImplementedError('Not Implemented GA for usual GP Bandits.')

  # Managing the GP ---------------------------------------------------------
  def _process_fit_gp(self, gp_fitter):
    """ Processes the results of gp_fitter.fit_gp(). We are using this in 2 places. """
    ret = gp_fitter.fit_gp()
    self.gp_processor = Namespace()
    self.gp_processor.fit_type = ret[0]
    self.gp = None # Mostly to avoid bugs
    if ret[0] == 'fitted_gp':
      self.gp_processor.fitted_gp = ret[1]
    elif ret[0] == 'sample_hps_with_probs':
      self.gp_processor.sample_hps = ret[1]
      self.gp_processor.gp_fitter = gp_fitter
      self.gp_processor.sample_probs = ret[2]
      use_hps_idxs = np.random.choice(len(ret[1]),
                                      size=(self.options.build_new_model_every,),
                                      replace=True,
                                      p=ret[2])
      self.gp_processor.use_hps = [ret[1][idx] for idx in use_hps_idxs]
    else:
      raise ValueError('Unknown option %s for results of fit_gp.'%(ret[0]))

  def _set_next_gp(self):
    """ Returns the next GP. """
    if not hasattr(self, 'gp_processor') or self.gp_processor is None:
      return
    if self.gp_processor.fit_type == 'fitted_gp':
      self.gp = self.gp_processor.fitted_gp
    elif self.gp_processor.fit_type == 'sample_hps_with_probs':
      next_gp_hps = self.gp_processor.use_hps.pop(0)
      self.gp_processor.use_hps.append(next_gp_hps)
      self.gp = self.gp_processor.gp_fitter.build_gp(next_gp_hps, build_posterior=False)
      reg_X, reg_Y = self._get_reg_X_reg_Y()
      self._child_set_gp_data(reg_X, reg_Y)
    if self.step_idx == self.last_model_build_at or \
       self.step_idx == self.last_model_build_at + 1:
      self._report_current_gp()

  def _child_set_gp_data(self, reg_X, reg_Y):
    """ Set data in child. Can be overridden by a child class. """
    self.gp.set_data(reg_X, reg_Y, build_posterior=True)

  def _child_build_new_model(self):
    """ Builds a new model. """
    self._build_new_gp()

  def _report_current_gp(self):
    """ Reports the current GP. """
    gp_fit_report_str = '    -- Fitting GP (j=%d): %s'%(self.step_idx, str(self.gp))
    self.reporter.writeln(gp_fit_report_str)

  def _get_reg_X_reg_Y(self):
    """ Returns the current data to be added to the GP. """
    reg_X = self.pre_eval_points + self.history.query_points
    reg_Y = np.concatenate((self.pre_eval_vals, self.history.query_vals), axis=0)
    return reg_X, reg_Y

  def _build_new_gp(self):
    """ Builds a GP with the data in history and stores in self.gp. """
    if hasattr(self.func_caller, 'init_gp') and self.func_caller.init_gp is not None:
      # If you know the true GP.
      raise NotImplementedError('Not implemented passing given GP yet.')
    else:
      if self.options.shrink_kernel_with_time:
        raise NotImplementedError('Not implemented kernel shrinking for the GP yet.')
      # Invoke the GP fitter.
      reg_X, reg_Y = self._get_reg_X_reg_Y()
      gp_fitter = self._get_gp_fitter(reg_X, reg_Y)
      self._process_fit_gp(gp_fitter)

  def _get_gp_fitter(self, reg_X, reg_Y):
    """ Returns a GP Fitter. Can be over-ridden by a child class. """
    return SimpleGPFitter(reg_X, reg_Y, options=self.options, reporter=self.reporter)

  def _child_add_data_to_model(self, qinfos):
    """ Add data to self.gp """
    if self.gp is None:
      return
    if len(qinfos) == 0:
      return
    new_points = []
    new_vals = np.empty(0)
    for i in range(len(qinfos)):
      new_points.append(qinfos[i].point)
      new_vals = np.append(new_vals, [qinfos[i].val], axis=0)
    self._add_data_to_gp(new_points, new_vals)

  def _add_data_to_gp(self, new_points, new_vals):
    """ Adds data to the GP. """
    # Add data to the GP only if we will be repeating with the same GP.
    if self.gp_processor.fit_type == 'fitted_gp':
      self.gp.add_data(new_points, new_vals)

  # Methods needed for initialisation ----------------------------------------
  def _child_optimise_initialise(self):
    """ No additional initialisation for GP bandit. """
#     self._create_init_gp()
    self._build_new_gp()

  def _create_init_gp(self):
    """ Creates an initial GP. """
    reg_X = self.pre_eval_points + self.history.query_points
    reg_Y = np.concatenate((self.pre_eval_vals, self.history.query_vals), axis=0)
    range_Y = reg_Y.max() - reg_Y.min()
    mean_func = lambda x: np.array([np.median(reg_Y)] * len(x))
    kernel = self.domain.get_default_kernel(range_Y)
    noise_var = (reg_Y.std()**2)/10
    self.gp = GP(reg_X, reg_Y, kernel, mean_func, noise_var)

  # Obtain the acquisition optimiser ---------------------------------------
  def _get_acq_optimise_func(self):
    """ Returns a function that can optimise the acquisition. """
    # pylint: disable=star-args
    acq_opt_method = self._get_acq_opt_method()
    if acq_opt_method in ['ga', 'rand_ga']:
      ret = lambda obj, max_evals: self.domain.maximise_obj(acq_opt_method,
                                     obj, max_evals, mutation_op=self.ga_mutation_op,
                                     init_pool=self.ga_init_pool)
    elif acq_opt_method in ['rand', 'direct']:
      ret = lambda obj, max_evals: self.domain.maximise_obj(acq_opt_method,
                                                            obj, max_evals)
    return ret

  # Methods needed for optimisation ----------------------------------------
  def _get_ancillary_data_for_acquisition(self):
    """ Returns ancillary data for the acquisitions. """
    max_num_acq_opt_evals = self.get_acq_opt_max_evals(self.step_idx)
    return Namespace(max_evals=max_num_acq_opt_evals,
                     t=self.step_idx,
                     curr_max_val=self.curr_opt_val,
                     evals_in_progress=self.eval_points_in_progress,
                     acq_opt_method=self.options.acq_opt_method)

  def _determine_next_eval_point(self):
    """ Determine the next point for evaluation. """
    anc_data = self._get_ancillary_data_for_acquisition()
    select_pt_func = getattr(gpb_acquisitions.asy, self.options.acq.lower())
    acq_optimise = self._get_acq_optimise_func()
    next_eval_point = select_pt_func(self.gp, acq_optimise, anc_data)
    return next_eval_point

  def _determine_next_batch_of_eval_points(self):
    """ Determine the next batch of eavluation points. """
    anc_data = self._get_ancillary_data_for_acquisition()
    select_pt_func = getattr(gpb_acquisitions.syn, self.options.acq.lower())
    acq_optimise = self._get_acq_optimise_func()
    next_batch_of_eval_points = select_pt_func(self.num_workers, self.gp,
                                               acq_optimise, anc_data)
    return next_batch_of_eval_points

  def _main_loop_pre(self):
    """ Things to be done before each iteration of the optimisation loop. """
    self._set_next_gp()


# GP Bandit class ends here
# =====================================================================================

# APIs for GP Bandit optimisation. ----------------------------------------------------

# 1. Optimisation from a FunctionCaller object.
def gpb_from_func_caller(func_caller, worker_manager, max_capital, mode=None, acq=None,
                         options=None, gp_args=None, reporter='default'):
  """ GP Bandit optimisation from a utils.function_caller.FunctionCaller instance. """
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(get_all_gp_bandit_args_from_gp_args(gp_args),
                           reporter=reporter)
  if acq is not None:
    options.acq = acq
  if mode is not None:
    options.mode = mode
  return (GPBandit(func_caller, worker_manager, options, reporter)).optimise(max_capital)

# # 2. Optimisation from all args.
# def gpb_from_args(func, domain_bounds, max_capital, acq=None, options=None,
#                   gp_args=None, reporter=None, vectorised=False, **kwargs):
#   """ This function executes GP Bandit (Bayesian) Optimisation.
#     Input Arguments:
#       - func: The function to be optimised.
#       - domain_bounds: The bounds for the domain.
#       - max_capital: The maximum capital for optimisation.
#       - options: A namespace which gives other options.
#       - reporter: A reporter object to write outputs.
#       - vectorised: If true, it means func take matrix inputs. If
#           false, they take only single point inputs.
#       - true_opt_pt, true_opt_val: The true optimum point and value (if known). Mostly
#           for experimenting with synthetic problems.
#       - time_distro: The time distribution to be used when sampling.
#       - time_distro_params: parameters for the time distribution.
#       - gp_args: Has arguments for the gp.
#     Returns: (gpb_opt_pt, gpb_opt_val, history)
#       - gpb_opt_pt, gpb_opt_val: The optimum point and value.
#       - history: A namespace which contains a history of all the previous queries.
#   """
#   func_caller = get_function_caller_from_function(func, domain_bounds=domain_bounds,
#                                                   vectorised=vectorised, **kwargs)
#   return gpb_from_func_caller(func_caller, max_capital, acq, options, gp_args, reporter)

