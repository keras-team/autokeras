"""
  GP based Bayesian Optimisation for Neural Networks.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=maybe-no-member


import numpy as np
# Local imports
from nas.nasbot.opt.blackbox_optimiser import blackbox_opt_args
from nas.nasbot.opt import gpb_acquisitions
from nas.nasbot.nn.nn_gp import nn_gp_args, NNGPFitter
from nas.nasbot.nn.nn_modifiers import get_nn_modifier_from_args
from nas.nasbot.nn.nn_comparators import get_default_otmann_distance
from nas.nasbot.opt.nn_opt_utils import get_initial_pool
from nas.nasbot.opt.gp_bandit import GPBandit, gp_bandit_args
from nas.nasbot.utils.general_utils import block_augment_array
from nas.nasbot.utils.reporters import get_reporter
from nas.nasbot.utils.option_handler import get_option_specs, load_options

nasbot_specific_args = [
  get_option_specs('nasbot_acq_opt_method', False, 'ga',
    'Which method to use when optimising the acquisition. Will override acq_opt_method' +
    ' in the arguments for gp_bandit.'),
  get_option_specs('ga_mutation_op_distro', False, 'd0.5-0.25-0.125-0.075-0.05',
    'Which method to use when optimising the acquisition. Will override acq_opt_method' +
    ' in the arguments for gp_bandit.'),
  ]

all_nasbot_args = nasbot_specific_args + gp_bandit_args + \
                        blackbox_opt_args + nn_gp_args
all_nn_random_bandit_args = all_nasbot_args

# NN GP Bandit Class --------------------------------------------------------------------
class NASBOT(GPBandit):
  """ NN GP Bandit. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, func_caller, worker_manager, tp_comp,
               options=None, reporter=None):
    """ Constructor.
        tp_comp: short for transport_distance_computer is a function that computes the
        otmann distances between two neural networks. Technically, it can be any distance
        but the rest of the code is implemented so as to pass an otmann distance computer.
    """
    # Set initial attributes
    self.tp_comp = tp_comp
    if options is None:
      reporter = get_reporter(reporter)
      options = load_options(all_nasbot_args, reporter=reporter)
    super(NASBOT, self).__init__(func_caller, worker_manager,
                                     options=options, reporter=reporter)

  def _child_set_up(self):
    """ Child up. """
    # First override the acquisition optisation method
    self.options.acq_opt_method = self.options.nasbot_acq_opt_method
    # No cal the super function
    super(NASBOT, self)._child_set_up()
    self.list_of_dists = None
    self.already_evaluated_dists_for = None
    # Create a GP fitter with no data and use its tp_comp as the bandit's tp_comp
    init_gp_fitter = NNGPFitter([], [], self.domain.get_type(), tp_comp=self.tp_comp,
                                list_of_dists=None, options=self.options,
                                reporter=self.reporter)
    self.tp_comp = init_gp_fitter.tp_comp
    self.mislabel_coeffs = init_gp_fitter.mislabel_coeffs
    self.struct_coeffs = init_gp_fitter.struct_coeffs

  def _set_up_acq_opt_ga(self):
    """ Determines the mutation operator for the internal GA. """
    # First the sampling distribution for the GA mutation operator
    mut_arg = self.options.ga_mutation_op_distro
    if isinstance(mut_arg, list):
      self.ga_mutation_op = get_nn_modifier_from_args(self.domain.constraint_checker,
                                                      dflt_num_steps_probs=mut_arg)
    elif isinstance(mut_arg, (int, int, float)):
      self.ga_mutation_op = get_nn_modifier_from_args(self.domain.constraint_checker,
                                                      dflt_max_num_steps=mut_arg)
    elif mut_arg.startswith('d'):
      ga_mutation_probs = [float(x) for x in mut_arg[1:].split('-')]
      self.ga_mutation_op = get_nn_modifier_from_args(self.domain.constraint_checker,
                                       dflt_num_steps_probs=ga_mutation_probs)
    elif mut_arg.startswith('n'):
      ga_mutation_num_steps = int(x[1:])
      self.ga_mutation_op = get_nn_modifier_from_args(self.domain.constraint_checker,
                                       dflt_max_num_steps=ga_mutation_num_steps)
    else:
      raise ValueError('Cannot parse ga_mutation_op_distro=%s.'%(
                       self.options.ga_mutation_op_distro))
    # The initial pool
    self.ga_init_pool = get_initial_pool(self.domain.get_type())
    # The number of evaluations
    if self.get_acq_opt_max_evals is None:
      lead_const = min(5, self.domain.get_dim())**2
      self.get_acq_opt_max_evals = lambda t: np.clip(
        lead_const * np.sqrt(t), 50, 500)

  def _compute_list_of_dists(self, X1, X2):
    """ Computes the list of distances. """
    return self.tp_comp(X1, X2, mislabel_coeffs=self.mislabel_coeffs,
                                struct_coeffs=self.struct_coeffs,
                                dist_type=self.options.dist_type)

  def _get_gp_fitter(self, reg_X, reg_Y):
    """ Builds a NN GP. """
    return NNGPFitter(reg_X, reg_Y, self.domain.get_type(), tp_comp=self.tp_comp,
                      list_of_dists=self.list_of_dists,
                      options=self.options,
                      reporter=self.reporter)

  def _add_data_to_gp(self, new_points, new_vals):
    """ Adds data to the GP. Also tracks list_of_dists. """
    # First add it to the list of distances
    if self.list_of_dists is None:
      # This is the first time, so use all the data.
      reg_X, _ = self._get_reg_X_reg_Y()
      self.list_of_dists = self._compute_list_of_dists(reg_X, reg_X)
      self.already_evaluated_dists_for = reg_X
    else:
      list_of_dists_old_new = self._compute_list_of_dists(
                                self.already_evaluated_dists_for, new_points)
      list_of_dists_new_new = self._compute_list_of_dists(new_points, new_points)
      self.already_evaluated_dists_for.extend(new_points)
      for idx in range(len(list_of_dists_old_new)):
        self.list_of_dists[idx] = block_augment_array(
          self.list_of_dists[idx], list_of_dists_old_new[idx],
          list_of_dists_old_new[idx].T, list_of_dists_new_new[idx])
    # Now add to the GP
    if self.gp_processor.fit_type == 'fitted_gp':
      self.gp.add_data(new_points, new_vals, build_posterior=False)
      self.gp.set_list_of_dists(self.list_of_dists)
      self.gp.build_posterior()

  def _child_set_gp_data(self, reg_X, reg_Y):
    """ Set Data for the child. """
    if self.list_of_dists is None:
      self.list_of_dists = self._compute_list_of_dists(reg_X, reg_X)
      self.already_evaluated_dists_for = reg_X
    if (len(reg_X), len(reg_Y)) != self.list_of_dists[0].shape:
      print (len(reg_X)), len(reg_Y), self.list_of_dists[0].shape, self.step_idx
    assert (len(reg_X), len(reg_Y)) == self.list_of_dists[0].shape
    self.gp.set_list_of_dists(self.list_of_dists)
    self.gp.set_data(reg_X, reg_Y, build_posterior=True)

# The random searcher ------------------------------------------------------------------
class NNRandomBandit(NASBOT):
  """ RandomNNBandit - uses the same search space as NASBOT but picks points randomly.
  """

  def __init__(self, func_caller, worker_manager, options=None, reporter=None):
    """ Constructor. """
    super(NNRandomBandit, self).__init__(func_caller, worker_manager,
                                         None, options, reporter)

  def _child_add_data_to_model(self, _):
    """ Adds data to the child data. """
    pass

  def _child_set_gp_data(self, reg_X, reg_Y):
    """ No GP to add child data to. """
    pass

  def _add_data_to_gp(self, new_points, new_vals):
    """ No GP to add child data to. """
    pass

  def _get_gp_fitter(self, reg_X, reg_Y):
    """ No need for this. """
    pass

  def _process_fit_gp(self, gp_fitter):
    """ No need for this. """
    pass

  def _set_next_gp(self):
    """ No need for this. """
    pass

  def _child_build_new_model(self):
    """ No need for this. """
    pass

  def _build_new_gp(self):
    """ No need for this. """
    pass

  def _create_init_gp(self):
    """ No need for this. """
    pass

  def _determine_next_eval_point(self):
    """ Here the acquisition we maximise will return random values. """
    anc_data = self._get_ancillary_data_for_acquisition()
    select_pt_func = gpb_acquisitions.asy.rand
    acq_optimise = self._get_acq_optimise_func()
    next_eval_point = select_pt_func(self.gp, acq_optimise, anc_data)
    return next_eval_point

  def _determine_next_batch_of_eval_points(self):
    """ Determine next batch. """
    anc_data = self._get_ancillary_data_for_acquisition()
    select_pt_func = gpb_acquisitions.syn.rand
    acq_optimise = self._get_acq_optimise_func()
    next_batch_of_eval_points = select_pt_func(self.num_workers, self.gp,
                                               acq_optimise, anc_data)
    return next_batch_of_eval_points


# APIs -----------------------------------------------------------------------------------
def nnrandbandit_from_func_caller(func_caller, worker_manager, max_capital,
                                  mode=None, options=None, reporter='default'):
  """ NNRandomBandit optimisation from a function caller. """
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(all_nn_random_bandit_args, reporter=reporter)
  if mode is not None:
    options.mode = mode
  options.acq = 'randnn'
  return (NNRandomBandit(func_caller, worker_manager, options=options,
                         reporter=reporter)).optimise(max_capital)

def nasbot(func_caller, worker_manager, budget, tp_comp=None,
           mode=None, init_pool=None, acq='hei', options=None, reporter='default'):
  """ NASBOT optimisation from a function caller. """
  nn_type = func_caller.domain.nn_type
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(all_nasbot_args, reporter=reporter)
  if acq is not None:
    options.acq = acq
  if mode is not None:
    options.mode = mode
  if tp_comp is None:
    tp_comp = get_default_otmann_distance(nn_type, 1.0)
  # Initial queries
  if not hasattr(options, 'pre_eval_points') or options.pre_eval_points is None:
    if init_pool is None:
      init_pool = get_initial_pool(nn_type)
    options.get_initial_points = lambda n: init_pool[:n]
  return (NASBOT(func_caller, worker_manager, tp_comp,
          options=options, reporter=reporter)).optimise(budget)

