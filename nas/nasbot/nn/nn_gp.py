"""..
  Implements a GP for neural networks. The kernels are taken from nn_comparators.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=abstract-class-not-used
# pylint: disable=arguments-differ

import numpy as np
# Local
from nas.nasbot.gp import gp_core
from nas.nasbot.gp.gp_instances import basic_gp_args
from nas.nasbot.nn import nn_comparators
from nas.nasbot.utils.ancillary_utils import get_list_of_floats_as_str
from nas.nasbot.utils.reporters import get_reporter
from nas.nasbot.utils.option_handler import get_option_specs, load_options

_DFLT_KERNEL_TYPE = 'lpemd_sum'

nn_gp_specific_args = [
  get_option_specs('dist_type', False, 'lp-emd',
    'The type of distance. This should be lp, emd or lp-emd.'),
# Use given coeffcients by default
  get_option_specs('choose_mislabel_struct_coeffs', False, 'use_given',
    ('How to choose the mislabel and struct coefficients. Should be one of ' +
     'tune_coeffs or use_given. In the latter case, mislabel_coeffs and struct_coeffs ' +
     'should be non-empty.')),
  get_option_specs('compute_kernel_from_dists', False, True,
    'Should you compute the kernel from pairwise distances whenever possible.'),
  get_option_specs('mislabel_coeffs', False, '1.0-1.0-1.0-1.0',
    'The mislabel coefficients specified as a string. If -1, it means we will tune.'),
  get_option_specs('struct_coeffs', False, '0.1-0.25-0.61-1.5',
    'The struct coefficients specified as a string. If -1, it means we will tune.'),
  get_option_specs('lp_power', False, 1,
    'The powers to use in the LP distance for the kernel.'),
  get_option_specs('emd_power', False, 2,
    'The powers to use in the EMD distance for the kernel.'),
  get_option_specs('non_assignment_penalty', False, 1.0,
    'The non-assignment penalty.'),
  ]

nn_gp_args = gp_core.mandatory_gp_args + basic_gp_args + nn_gp_specific_args


# Main GP class for neural network architectures -----------------------------------------
class NNGP(gp_core.GP):
  """ A Gaussian process for Neural Networks. """

  def __init__(self, X, Y, kernel_type, nn_type,
               kernel_hyperparams, mean_func, noise_var,
               dist_type, tp_comp=None, list_of_dists=None, *args, **kwargs):
    # pylint: disable=too-many-arguments
    """ Constructor.
        kernel_type: Should be one of 'from_distance_computer', 'lp', 'emd', 'sum_lp-emd',
                     'product_lp-emd'
        kernel_hyperparams: is a python dictionary specifying the hyper-parameters for the
                            kernel. Will have parameters 'beta', 'scale', 'sum_scales',
                            'powers'.
        list_of_dists: Is a list of distances for the training set.
        tp_comp: a transport distance computer object.
    """
    kernel = self._get_kernel_from_type(kernel_type, nn_type, kernel_hyperparams,
                                        tp_comp, dist_type)
    self.nn_type = nn_type
    self.list_of_dists = list_of_dists
    # Call super constructor
    super(NNGP, self).__init__(X, Y, kernel, mean_func, noise_var,
                               handle_non_psd_kernels='project_first', *args, **kwargs)

  def build_posterior(self):
    """ Checks if the sizes of list of distances and the data are consistent before
        calling build_posterior from the super class.
    """
    if self.list_of_dists is not None:
      assert self.list_of_dists[0].shape == (len(self.X), len(self.Y))
    super(NNGP, self).build_posterior()

  def set_list_of_dists(self, list_of_dists):
    """ Updates the list of distances. """
    self.list_of_dists = list_of_dists

  def _get_training_kernel_matrix(self):
    """ Compute the kernel matrix from distances if they are provided. """
    if self.list_of_dists is not None:
      return self.kernel.evaluate_from_dists(self.list_of_dists)
    else:
      return self.kernel(self.X, self.X)

  def _child_str(self):
    """ Description of the child GP. """
    scales_str = self._get_scales_str()
    betas_str = 'betas=' + get_list_of_floats_as_str(self.kernel.hyperparams['betas'])
    return self.kernel.dist_type + ':: ' + scales_str + ' ' + betas_str

  def _get_scales_str(self):
    """ Returns a description of the alphas. """
    if 'alphas' in self.kernel.hyperparams.keys():
      return 'alphas=' + get_list_of_floats_as_str(self.kernel.hyperparams['alphas'])
    else:
      return 'scale=' + str(round(self.kernel.hyperparams['scale'], 4))

  @classmethod
  def _get_kernel_from_type(cls, kernel_type, nn_type, kernel_hyperparams,
                            tp_comp, dist_type):
    """ Returns the kernel. """
    if tp_comp is None:
      tp_comp = nn_comparators.get_otmann_distance_from_args(nn_type,
                  kernel_hyperparams['non_assignment_penalty'])
    # Now construct the kernel.
    if kernel_type in ['lpemd_prod', 'lp', 'emd']:
      return nn_comparators.DistProdNNKernel(tp_comp,
              kernel_hyperparams['mislabel_coeffs'],
              kernel_hyperparams['struct_coeffs'],
              kernel_hyperparams['betas'],
              kernel_hyperparams['scale'],
              kernel_hyperparams['powers'],
              dist_type)
    elif kernel_type in ['lpemd_sum']:
      return nn_comparators.DistSumNNKernel(tp_comp,
              kernel_hyperparams['mislabel_coeffs'],
              kernel_hyperparams['struct_coeffs'],
              kernel_hyperparams['alphas'],
              kernel_hyperparams['betas'],
              kernel_hyperparams['powers'],
              dist_type)
    else:
      raise ValueError('Unknown kernel_type %s.'%(kernel_type))


# GP Fitter for neural network architectures ---------------------------------------------
class NNGPFitter(gp_core.GPFitter):
  """ Fits a GP by tuning the kernel hyper-params. """
  # pylint: disable=attribute-defined-outside-init

  def __init__(self, X, Y, nn_type, tp_comp=None, list_of_dists=None,
               options=None, reporter=None, *args, **kwargs):
    """ Constructor. """
    self.X = X
    self.Y = Y
    self.nn_type = nn_type
    self.reporter = get_reporter(reporter)
    self.list_of_dists = list_of_dists
    self.num_data = len(X)
    self.tp_comp = tp_comp
    if options is None:
      options = load_options(nn_gp_args, 'GPFitter', reporter=reporter)
    super(NNGPFitter, self).__init__(options, *args, **kwargs)

  def _child_set_up(self):
    """ Sets up child-specific parameters for the GP Fitter. """
    # pylint: disable=too-many-branches
    # Check args
    if not self.options.dist_type in ['lp', 'emd', 'lp-emd']:
      raise ValueError('Unknown dist_type. Should be one of \'lp\', \'emd\', \'lp-emd\'.')
    if not self.options.kernel_type in ['lp', 'emd', 'lpemd_sum', 'lpemd_prod',
                                        'default']:
      raise ValueError('Unknown dist_type. Should be one of \'lp\', \'emd\', ' +
                       '\'lpemd_sum\', or \'lpemd_prod\'.')
    if not self.options.noise_var_type in ['tune', 'label', 'value']:
      raise ValueError('Unknown noise_var_type. Should be either tune, label or value.')
    if not self.options.mean_func_type in ['mean', 'median', 'const', 'zero']:
      raise ValueError('Unknown mean_func_type. Should be one of mean/median/const/zero.')
    if not self.options.choose_mislabel_struct_coeffs in ['use_given', 'tune_coeffs']:
      raise ValueError('Unknown value for choose_mislabel_struct_coeffs. Should be one ' +
                       'of use_given or tune_coeffs.')
    # Set kernel type
    if self.options.kernel_type == 'default':
      self.kernel_type = _DFLT_KERNEL_TYPE
    else:
      self.kernel_type = self.options.kernel_type
    # Check if dist_type and kernel_type are consistent
    if self.options.dist_type in ['lp', 'emd'] and \
       self.kernel_type != self.options.dist_type:
      raise ValueError('If dist_type is %s, then kernel_type should be %s.'%(
                       self.options.dist_type, self.options.dist_type))
    elif self.options.dist_type == 'lp-emd' and \
      self.kernel_type not in ['lpemd_prod', 'lpemd_sum']:
      raise ValueError('If dist_type is lp-emd, then kernel_type should be lpemd_sum or '
                       + 'lpemd_prod.')
    if self.options.choose_mislabel_struct_coeffs == 'use_given' and \
      (self.options.mislabel_coeffs == '' or self.options.struct_coeffs == ''):
      raise ValueError(('If choose_mislabel_struct_coeffs is use_given, then ' +
        'mislabel_coeffs and struct_coeffs cannot be empty. Given mislabel_coeffs=%s' +
        ' struct_coeffs=%s')%(self.options.mislabel_coeffs, self.options.struct_coeffs))
    self._preprocess_struct_mislabel_coeffs()
    # Check if we need to pre-compute distances
    if self.options.choose_mislabel_struct_coeffs == 'use_given' and \
       self.list_of_dists is None:
      if self.tp_comp is None:
        self.tp_comp = nn_comparators.get_otmann_distance_from_args(self.nn_type,
                                        self.options.non_assignment_penalty)
      self.list_of_dists = self.tp_comp(self.X, self.X, self.mislabel_coeffs,
                                        self.struct_coeffs, self.options.dist_type)
    # Some parameters we will be using often
    if len(self.Y) > 1:
      self.Y_var = self.Y.std() ** 2
    else:
      self.Y_var = 1.0
    # Compute the pairwise distances here
    # Set up based on whether we are doing maximum likelihood or post_sampling
    if self.options.hp_tune_criterion == 'ml':
      self.hp_bounds = []
      self._child_set_up_ml_tune()
    elif self.options.hp_tune_criterion == 'post_sampling':
      self.hp_priors = []
      self._child_set_up_post_sampling()

  def _preprocess_struct_mislabel_coeffs(self):
    """ Preprocesses the structural and mislabel coefficients. """
    if self.options.choose_mislabel_struct_coeffs == 'tune_coeffs':
      # If they are integers or floats
      self.num_mislabel_struct_coeffs = 1
      self.to_tune_mislabel_struct_coeffs = True
    elif (isinstance(self.options.mislabel_coeffs, list) and
          isinstance(self.options.struct_coeffs, list)) or \
         (isinstance(self.options.mislabel_coeffs, str) and
          isinstance(self.options.struct_coeffs, str)):
      # If they are lists or strings
      if isinstance(self.options.mislabel_coeffs, str):
        # If they are strings, obtain the lists
        self.mislabel_coeffs = [float(x) for x in self.options.mislabel_coeffs.split('-')]
        self.struct_coeffs = [float(x) for x in self.options.struct_coeffs.split('-')]
      else:
        self.mislabel_coeffs = self.options.mislabel_coeffs
        self.struct_coeffs = self.options.struct_coeffs
      # Store other parameters
      self.num_mislabel_struct_coeffs = len(self.mislabel_coeffs)
      self.to_tune_mislabel_struct_coeffs = False
      # Check if the lengths are the same
      if len(self.mislabel_coeffs) != len(self.struct_coeffs):
        raise ValueError('Length of mislabel and structural coefficients must be same.' +
                         'Given: mislabel: %s (%d), struct:%s (%d).'%(
                         str(self.options.mislabel_coeffs), len(self.mislabel_coeffs),
                         str(self.options.struct_coeffs), len(self.struct_coeffs)))
    else:
      raise ValueError('Bad format for mislabel and structural coefficients.' +
                       'Given: mislabel: %s, struct:%s.'%(
                      str(self.options.mislabel_coeffs), str(self.options.struct_coeffs)))

  def _child_set_up_ml_tune(self):
    """ Sets up tuning for Maximum likelihood. """
    # Order of hyper-parameters (this is critical): noies variance, scale/alphas, betas,
    # mislabel_coeff, struct_coeff
    # 1. Noise variance ---------------------------------------
    if self.options.noise_var_type == 'tune':
      self.noise_var_log_bounds = [np.log(0.005 * self.Y_var), np.log(0.2 * self.Y_var)]
      self.hp_bounds.append(self.noise_var_log_bounds)
    # 2. scale/alphas -----------------------------------------
    scale_log_bounds = [np.log(0.01 * self.Y_var), np.log(100 * self.Y_var)]
    if self.kernel_type in ['lp', 'emd', 'lpemd_prod']:
      self.hp_bounds.append(scale_log_bounds)
    elif self.kernel_type == 'lpemd_sum':
      self.hp_bounds.extend([scale_log_bounds] * 2)
    # 3. betas ------------------------------------------------
    lp_beta_log_bounds = [[np.log(1e-9), np.log(1e-3)]] * self.num_mislabel_struct_coeffs
    emd_beta_log_bounds = [[np.log(1e-1), np.log(1e2)]] * self.num_mislabel_struct_coeffs
    if self.options.dist_type == 'lp':
      all_beta_bounds = lp_beta_log_bounds
    elif self.options.dist_type == 'emd':
      all_beta_bounds = emd_beta_log_bounds
    elif self.options.dist_type == 'lp-emd':
      all_beta_bounds = [j for i in zip(lp_beta_log_bounds, emd_beta_log_bounds)
                         for j in i]
    self.hp_bounds.extend(all_beta_bounds) # extend with log_beta_bounds
    # 4 & 5. mislabel/struct_coeff ----------------------------
    if self.to_tune_mislabel_struct_coeffs:
      self.hp_bounds.append([0.001, 2.0]) # mislabel coefficient (not in log space)
      self.hp_bounds.append([0.001, 2.0]) # structural coefficient (not in log space)

  def _child_set_up_post_sampling(self):
    """ Sets up posterior sampling. """
    raise NotImplementedError('Not implemented post sampling yet.')

  def _child_build_gp(self, gp_hyperparams, build_posterior=True):
    """ Builds the GP from the hyper-parameters. """
    gp_hyperparams = gp_hyperparams[:] # create a copy of the list
    kernel_hyperparams = {}
    # mean function
    mean_func = gp_core.get_mean_func_from_options(self.options, self.Y) # Mean function
    # Extract GP hyper-parameters
    # 1. Noise variance ------------------------------------------------------
    noise_var = gp_core.get_noise_var_from_options_and_hyperparams(self.options,
                  gp_hyperparams, self.Y, 0)
    if self.options.noise_var_type == 'tune':
      gp_hyperparams = gp_hyperparams[1:]
    # 2. Scale and alphas
    if self.kernel_type in ['lp', 'emd', 'lpemd_prod']:
      kernel_hyperparams['scale'] = np.exp(gp_hyperparams[0])
      gp_hyperparams = gp_hyperparams[1:]
    elif self.kernel_type == 'lpemd_sum':
      kernel_hyperparams['alphas'] = np.exp(gp_hyperparams[:2])
      gp_hyperparams = gp_hyperparams[2:]
    # 3. betas
    if self.options.dist_type in ['lp', 'emd']:
      kernel_hyperparams['betas'] = np.exp(
                                    gp_hyperparams[:self.num_mislabel_struct_coeffs])
      gp_hyperparams = gp_hyperparams[self.num_mislabel_struct_coeffs:]
    elif self.options.dist_type == 'lp-emd':
      kernel_hyperparams['betas'] = np.exp(
                                    gp_hyperparams[:2 * self.num_mislabel_struct_coeffs])
      gp_hyperparams = gp_hyperparams[2 * self.num_mislabel_struct_coeffs:]
    # 4 & 5. mislabel and structural coefficents
    if self.to_tune_mislabel_struct_coeffs:
      kernel_hyperparams['mislabel_coeffs'] = gp_hyperparams[0]
      kernel_hyperparams['struct_coeffs'] = gp_hyperparams[1]
      gp_hyperparams = gp_hyperparams[2:]
    else:
      kernel_hyperparams['mislabel_coeffs'] = self.mislabel_coeffs
      kernel_hyperparams['struct_coeffs'] = self.struct_coeffs
    # Some checks
    assert len(gp_hyperparams) == 0
    # Build the remainder of the kernel ---------------------------------
    # - powers
    lp_powers = [self.options.lp_power] * self.num_mislabel_struct_coeffs
    emd_powers = [self.options.emd_power] * self.num_mislabel_struct_coeffs
    if self.options.dist_type == 'lp':
      kernel_hyperparams['powers'] = lp_powers
    elif self.options.dist_type == 'emd':
      kernel_hyperparams['powers'] = emd_powers
    elif self.options.dist_type == 'lp-emd':
      kernel_hyperparams['powers'] = [j for i in zip(lp_powers, emd_powers) for j in i]
    # - non_assignment_penalty
    kernel_hyperparams['non_assignment_penalty'] = self.options.non_assignment_penalty
    # Build the GP and return
    return NNGP(self.X, self.Y, self.kernel_type, self.nn_type,
                kernel_hyperparams, mean_func, noise_var,
                self.options.dist_type, tp_comp=self.tp_comp,
                list_of_dists=self.list_of_dists,
                build_posterior=build_posterior)

