"""
  A module for fitting a GP and tuning its kernel.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=abstract-class-not-used

import sys
import numpy as np
from scipy.linalg import solve_triangular
# Local imports
from nas.nasbot.utils.general_utils import stable_cholesky, draw_gaussian_samples, \
                                project_symmetric_to_psd_cone
from nas.nasbot.utils.oper_utils import direct_ft_maximise, random_maximise, random_sample
from nas.nasbot.utils.option_handler import get_option_specs, load_options
from nas.nasbot.utils.reporters import get_reporter


# These are mandatory requirements. Every GP implementation should probably use them.
mandatory_gp_args = [
  get_option_specs('hp_tune_criterion', False, 'ml',
                   'Which criterion to use when tuning hyper-parameters. Other ' +
                   'options are post_sampling and post_mean.'),
  get_option_specs('ml_hp_tune_opt', False, 'direct',
                   'Which optimiser to use when maximising the tuning criterion.'),
  get_option_specs('hp_tune_max_evals', False, -1,
                   'How many evaluations to use when maximising the tuning criterion.'),
  get_option_specs('handle_non_psd_kernels', False, '',
                   'How to handle kernels that are non-psd.')
  ]

def _solve_lower_triangular(A, b):
  """ Solves Ax=b when A is lower triangular. """
  return solve_triangular(A, b, lower=True)

def _solve_upper_triangular(A, b):
  """ Solves Ax=b when A is upper triangular. """
  return solve_triangular(A, b, lower=False)

def _check_feature_label_lengths_and_format(X, Y):
  """ Checks if the length of X and Y are the same. """
  if not isinstance(X, np.ndarray):
    return
  if X.shape[0] != len(Y):
    raise ValueError('Size of X (' + str(X.shape) + ') and Y (' +
      str(Y.shape) + ') do not match.')
  if len(X.shape) != 2 or len(Y.shape) != 1:
    raise ValueError('X should be an nxd matrix and Y should be an n-vector.' +
      'Given shapes of X, Y are: ', str(X.shape) + ', ' + str(Y.shape))


class GP(object):
  '''
  Base class for Gaussian processes.
  '''
  def __init__(self, X, Y, kernel, mean_func, noise_var, build_posterior=True,
               reporter=None, handle_non_psd_kernels=''):
    """ Constructor. """
    super(GP, self).__init__()
    _check_feature_label_lengths_and_format(X, Y)
    self.X = X
    self.Y = Y
    self.kernel = kernel
    self.mean_func = mean_func
    self.noise_var = noise_var
    self.reporter = get_reporter(reporter)
    self.handle_non_psd_kernels = handle_non_psd_kernels
    # Some derived attribues.
    self.num_tr_data = len(self.Y)
    # Initialise other attributes we will need.
    self.L = None
    self.alpha = None
    self.K_trtr = None
    self.K_trtr_wo_noise = None
    self._set_up()
    self.in_debug_mode = False
    # Build posterior if necessary
    if build_posterior:
      self.build_posterior()

  def _set_up(self):
    """ Additional set up. """
    # Test that if the kernel is not guaranteed to be PSD, then have have a way to
    # handle it.
    if not self.kernel.is_guaranteed_psd():
      assert self.handle_non_psd_kernels in ['project_first',
                                                     'try_before_project']

  def _write_message(self, msg):
    """ Writes a message via the reporter or the std out. """
    if self.reporter:
      self.reporter.write(msg)
    else:
      sys.stdout.write(msg)

  def set_data(self, X, Y, build_posterior=True):
    """ Sets the data to X and Y. """
    self.X = X
    self.Y = Y
    self.num_tr_data = len(self.Y)
    if build_posterior:
      self.build_posterior()

  def add_data(self, X_new, Y_new, build_posterior=True):
    """ Adds new data to the GP.
        If build_posterior is true it build_posteriors the posterior. """
    _check_feature_label_lengths_and_format(X_new, Y_new)
    self.X = self._append_X_data(self.X, X_new)
    self.Y = np.append(self.Y, Y_new)
    self.num_tr_data = len(self.Y)
    self.in_debug_mode = True
    if build_posterior:
      self.build_posterior()

  @classmethod
  def _append_X_data(cls, X1, X2):
    """ Appends X2 to X1 depending on what type the input is. Can be over-ridden by
        a child class. """
    if isinstance(X1, np.ndarray) and isinstance(X2, np.ndarray):
      return np.vstack((X1, X2))
    elif isinstance(X1, list) and isinstance(X2, list):
      return X1 + X2
    else:
      raise NotImplementedError('Append can only handle numpy arrays and list.' +
                                'For other types, over-ride this method in child class.')

  def build_posterior(self):
    """ Builds the posterior GP by computing the mean and covariance. """
    self.K_trtr_wo_noise = self._get_training_kernel_matrix()
    self.L = _get_cholesky_decomp(self.K_trtr_wo_noise, self.noise_var,
                                  self.handle_non_psd_kernels)
#     try:
#       self.L = _get_cholesky_decomp(self.K_trtr_wo_noise, self.noise_var,
#                                     self.handle_non_psd_kernels)
#     except Exception as e:
#       print e
#       import pdb
#       pdb.set_trace()
    Y_centred = self.Y - self.mean_func(self.X)
    self.alpha = _solve_upper_triangular(self.L.T,
                                         _solve_lower_triangular(self.L, Y_centred))

  def _get_training_kernel_matrix(self):
    """ Returns the training kernel matrix. Writing this as a separate method in case,
        the kernel computation can be done efficiently for a child class.
    """
    return self.kernel(self.X, self.X)

  def eval(self, X_test, uncert_form='none'):
    """ Evaluates the GP on X_test. If uncert_form is
          covar: returns the entire covariance on X_test (nxn matrix)
          std: returns the standard deviations on the test set (n vector)
          none: returns nothing (default).
    """
    # First check for uncert_form
    if not uncert_form in ['none', 'covar', 'std']:
      raise ValueError('uncert_form should be one of none, std or covar.')
    # Compute the posterior mean.
    test_mean = self.mean_func(X_test)
    K_tetr = self.kernel(X_test, self.X)
    pred_mean = test_mean + K_tetr.dot(self.alpha)
    # Compute the posterior variance or standard deviation as required.
    if uncert_form == 'none':
      uncert = None
    else:
      K_tete = self.kernel(X_test, X_test)
      V = _solve_lower_triangular(self.L, K_tetr.T)
      post_covar = K_tete - V.T.dot(V)
      post_covar = _get_post_covar_from_raw_covar(post_covar, self.noise_var,
                                                  self.kernel.is_guaranteed_psd())
      if uncert_form == 'covar':
        uncert = post_covar
      elif uncert_form == 'std':
        uncert = np.sqrt(np.diag(post_covar))
      else:
        raise ValueError('uncert_form should be none, covar or std.')
    return (pred_mean, uncert)

  def eval_with_hallucinated_observations(self, X_test, X_halluc, uncert_form='none'):
    """ Evaluates the GP with additional hallucinated observations in the
        kernel matrix. """
    if len(X_halluc) == 0:
      return self.eval(X_test, uncert_form)
    pred_mean, _ = self.eval(X_test, uncert_form='none') # Just compute the means.
    if uncert_form == 'none':
      uncert = None
    else:
      # Computed the augmented kernel matrix and its cholesky decomposition.
      X_aug = self._append_X_data(self.X, X_halluc)
      K_haha = self.kernel(X_halluc, X_halluc) # kernel for the hallucinated data
      K_trha = self.kernel(self.X, X_halluc)
      aug_K_trtr_wo_noise = np.vstack((np.hstack((self.K_trtr_wo_noise, K_trha)),
                              np.hstack((K_trha.T, K_haha))))
      aug_L = _get_cholesky_decomp(aug_K_trtr_wo_noise, self.noise_var,
                                   self.handle_non_psd_kernels)
      # Augmented kernel matrices for the test data
      aug_K_tete = self.kernel(X_test, X_test)
      aug_K_tetr = self.kernel(X_test, X_aug)
      aug_V = _solve_lower_triangular(aug_L, aug_K_tetr.T)
      aug_post_covar = aug_K_tete - aug_V.T.dot(aug_V)
      aug_post_covar = _get_post_covar_from_raw_covar(aug_post_covar, self.noise_var,
                                                      self.kernel.is_guaranteed_psd())
      if uncert_form == 'covar':
        uncert = aug_post_covar
      elif uncert_form == 'std':
        uncert = np.sqrt(np.diag(aug_post_covar))
      else:
        raise ValueError('uncert_form should be none, covar or std.')
    return (pred_mean, uncert)

  def compute_log_marginal_likelihood(self):
    """ Computes the log marginal likelihood. """
    Y_centred = self.Y - self.mean_func(self.X)
    ret = -0.5 * Y_centred.T.dot(self.alpha) - (np.log(np.diag(self.L))).sum() \
          - 0.5 * self.num_tr_data * np.log(2*np.pi)
    return ret

  def __str__(self):
    """ Returns a string representation of the GP. """
    return '%s, eta2: %0.4f (n=%d)'%(self._child_str(), self.noise_var, len(self.Y))

  def _child_str(self):
    """ String representation for child GP. """
    raise NotImplementedError('Implement in child class. !')

  def draw_samples(self, num_samples, X_test=None, mean_vals=None, covar=None):
    """ Draws num_samples samples at returns their values at X_test. """
    if X_test is not None:
      mean_vals, covar = self.eval(X_test, 'covar')
    return draw_gaussian_samples(num_samples, mean_vals, covar)

  def draw_samples_with_hallucinated_observations(self, num_samples, X_test,
                                                  X_halluc):
    """ Draws samples with hallucinated observations. """
    mean_vals, aug_covar = self.eval_with_hallucinated_observations(X_test,
                         X_halluc, uncert_form='covar')
    return draw_gaussian_samples(num_samples, mean_vals, aug_covar)


class GPFitter(object):
  """
    Class for fitting Gaussian processes.
  """
  # pylint: disable=attribute-defined-outside-init
  # pylint: disable=abstract-class-not-used
  # pylint: disable=arguments-differ

  def __init__(self, options, reporter='default'):
    """ Constructor. """
    super(GPFitter, self).__init__()
    self.reporter = get_reporter(reporter)
    if isinstance(options, list):
      options = load_options(options, 'GP', reporter=self.reporter)
    self.options = options
    self._set_up()

  def _set_up(self):
    """ Sets up a bunch of ancillary parameters. """
    # The following hyper-parameters need to be set mandatorily in _child_setup.
    self.hp_bounds = None # The bounds for each hyper parameter should be a num_hps x 2
                          # array where the 1st/2nd columns are the lowe/upper bounds.
    # Set up hyper-parameters for the child.
    self._child_set_up()
    # Some post child set up
    if self.options.hp_tune_criterion == 'ml':
      self.hp_bounds = np.array(self.hp_bounds)
      self.num_hps = len(self.hp_bounds) # The number of hyper parameters
      self._set_up_ml_hp_tune()
    elif self.options.hp_tune_criterion == 'post_sampling':
      self.num_hps = len(self.hp_priors) # The number of hyper parameters
      self._set_up_post_sampling_hp_tune()
    elif self.options.hp_tune_criterion == 'post_mean':
      self._set_up_post_mean_hp_tune()
    else:
      raise ValueError('hp_tune_criterion should be ml or post_sampling.')

  def _child_set_up(self):
    """ Here you should set up parameters for the child, such as the bounds for the
        optimiser etc. """
    raise NotImplementedError('Implement _child_set_up in a child method.')

  def _set_up_ml_hp_tune(self):
    """ Sets up optimiser for direct. """
    # define the following internal functions to abstract things out more.
    def _direct_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      _, opt_pt, _ = direct_ft_maximise(*args)
      return opt_pt
    def _rand_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      _, opt_pt = random_maximise(*args, vectorised=False)
      return opt_pt
    def _rand_exp_sampling_wrap(*args):
      """ A wrapper so as to only return the optimal point. """
      sample_hps, lml_vals = random_sample(*args, vectorised=False)
#       sample_probs = np.exp(lml_vals/np.sqrt(self.num_data))
      sample_probs = np.exp(lml_vals)
      sample_probs = sample_probs / sample_probs.sum()
      return sample_hps, sample_probs
    # Set some parameters
    if (hasattr(self.options, 'hp_tune_max_evals') and
        self.options.hp_tune_max_evals is not None and
        self.options.hp_tune_max_evals > 0):
      hp_tune_max_evals = self.options.hp_tune_max_evals
    else:
      hp_tune_max_evals = None
    # Set hp_optimise
    if self.options.ml_hp_tune_opt == 'direct':
      lower_bounds = self.hp_bounds[:, 0]
      upper_bounds = self.hp_bounds[:, 1]
      if hp_tune_max_evals is None:
        hp_tune_max_evals = min(1e4, max(500, self.num_hps * 50))
      self.hp_optimise = lambda obj: _direct_wrap(obj, lower_bounds, upper_bounds,
                                                  hp_tune_max_evals)
    elif self.options.ml_hp_tune_opt == 'rand':
      if hp_tune_max_evals is None:
        hp_tune_max_evals = min(1e4, max(500, self.num_hps * 200))
      self.hp_optimise = lambda obj: _rand_wrap(obj, self.hp_bounds,
                                                hp_tune_max_evals)
    elif self.options.ml_hp_tune_opt == 'rand_exp_sampling':
      if hp_tune_max_evals is None:
        hp_tune_max_evals = min(1e5, max(500, self.num_hps * 400))
      self.hp_sampler = lambda obj: _rand_exp_sampling_wrap(obj, self.hp_bounds,
                                                            hp_tune_max_evals)

  def _set_up_post_sampling_hp_tune(self):
    """ Sets up posterior sampling for tuning the parameters of the GP. """
    raise NotImplementedError('Not implemented posterior sampling yet.')

  def _set_up_post_mean_hp_tune(self):
    """ Sets up posterior sampling for tuning the parameters of the GP. """
    raise NotImplementedError('Not implemented posterior mean yet.')

  def build_gp(self, gp_hyperparams, *args, **kwargs):
    """ A method which builds a GP from the given gp_hyperparameters. It calls
        _child_build_gp after running some checks. """
    # Check the length of the hyper-parameters
    if self.num_hps != len(gp_hyperparams):
      raise ValueError('gp_hyperparams should be of length %d. Given length: %d.'%(
        self.num_hps, len(gp_hyperparams)))
    return self._child_build_gp(gp_hyperparams, *args, **kwargs)

  def _child_build_gp(self, gp_hyperparams, build_posterior):
    """ A method which builds the child GP from the given gp_hyperparameters. Should be
        implemented in a child method. """
    raise NotImplementedError('Implement _build_gp in a child method.')

  def _tuning_objective(self, gp_hyperparams):
    """ This function computes the tuning objective (such as the marginal likelihood)
        which is to be maximised in fit_gp. """
    built_gp = self.build_gp(gp_hyperparams)
    if self.options.hp_tune_criterion in ['ml', 'marginal_likelihood']:
      ret = built_gp.compute_log_marginal_likelihood()
    elif self.options.hp_tune_criterion in ['cv', 'cross_validation']:
      raise NotImplementedError('Yet to implement cross validation based hp-tuning.')
    else:
      raise ValueError('hp_tune_criterion should be either ml or cv')
    return ret

  def fit_gp(self):
    """ Fits a GP according to the tuning criterion. Returns the best GP along with the
        hyper-parameters. """
    if self.options.hp_tune_criterion == 'ml':
      if self.options.ml_hp_tune_opt in ['direct', 'rand']:
        opt_hps = self.hp_optimise(self._tuning_objective)
        opt_gp = self.build_gp(opt_hps)
        return 'fitted_gp', opt_gp, opt_hps
      elif self.options.ml_hp_tune_opt == 'rand_exp_sampling':
        sample_hps, sample_probs = self.hp_sampler(self._tuning_objective)
        return 'sample_hps_with_probs', sample_hps, sample_probs
  # GPFitter class ends here ------------------------------------------------------------


# Some utilities we will be using above -------------------------------------------------
def _get_cholesky_decomp(K_trtr_wo_noise, noise_var, handle_non_psd_kernels):
  """ Computes cholesky decomposition after checking how to handle non-psd kernels. """
  if handle_non_psd_kernels == 'try_before_project':
    K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
    try:
      # If the cholesky decomposition on the (noise added) matrix, return!
      L = stable_cholesky(K_trtr_w_noise, add_to_diag_till_psd=False)
      return L
    except np.linalg.linalg.LinAlgError:
      # otherwise, project and return
      return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, 'project_first')
  elif handle_non_psd_kernels == 'project_first':
    # project the Kernel (without noise) to the PSD cone and return
    K_trtr_wo_noise = project_symmetric_to_psd_cone(K_trtr_wo_noise)
    return _get_cholesky_decomp(K_trtr_wo_noise, noise_var, '')
  elif handle_non_psd_kernels == '':
    K_trtr_w_noise = K_trtr_wo_noise + noise_var * np.eye(K_trtr_wo_noise.shape[0])
    return stable_cholesky(K_trtr_w_noise)
  else:
    raise ValueError('Unknown option for handle_non_psd_kernels: %s'%(
                     handle_non_psd_kernels))

def _get_post_covar_from_raw_covar(raw_post_covar, noise_var, is_guaranteed_psd):
  """ Computes the posterior covariance from the raw_post_covar. This is mostly to
      account for the fact that the kernel may not be psd.
  """
  if is_guaranteed_psd:
    return raw_post_covar
  else:
    epsilon = 0.05 * noise_var
    return project_symmetric_to_psd_cone(raw_post_covar, epsilon=epsilon)



# Functions to be used by child class ====================================================
def get_mean_func_from_options(options, Y):
  """ Returns a function handle which is to be used as the mean function.
      options has attributes mean_func, mean_func_type and mean_func_const_value while Y
      is an array of the obervations.
  """
  # Mean function #####################################
  if hasattr(options, 'mean_func') and options.mean_func is not None:
    mean_func = options.mean_func
  else:
    if options.mean_func_type == 'mean':
      mean_func_const_value = Y.mean()
    elif options.mean_func_type == 'median':
      mean_func_const_value = np.median(Y)
    elif options.mean_func_type == 'const':
      mean_func_const_value = options.mean_func_const
    else:
      mean_func_const_value = 0
    mean_func = lambda x: np.array([mean_func_const_value] * len(x))
  return mean_func

def get_noise_var_from_options_and_hyperparams(options, gp_hyperparams, Y,
                                               noise_var_idx=0):
  """ Returns the noise variance from options and returns.
      options has attributes noise_var_type, noise_var_label and noise_var_value.
      noise_var_idx is the index noise_var is stored in gp_hyperparams.
  """
  if options.noise_var_type == 'tune':
    noise_var = np.exp(gp_hyperparams[noise_var_idx])
  elif options.noise_var_type == 'label':
    noise_var = options.noise_var_label * (Y.std()**2)
  else:
    noise_var = options.noise_var_value
  return noise_var

