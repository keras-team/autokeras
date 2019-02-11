"""
  Test cases for nn_gpy.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name

import numpy as np
# Local imports
from ..nn.syn_nn_functions import syn_func1_common, cnn_syn_func1, mlp_syn_func1
from ..nn.nn_examples import generate_cnn_architectures, generate_mlp_architectures
from ..nn import nn_gp
from ..utils.ancillary_utils import get_list_of_floats_as_str
from ..utils.base_test_class import BaseTestClass, execute_tests
from ..utils.option_handler import load_options

_TOL = 1e-5

# Create data ---------------------------------------------------------------------------
def gen_gp_test_data():
  """ Generates test data for the unit tests. """
  # CNNs
  cnns = generate_cnn_architectures()
  np.random.shuffle(cnns)
  X1_tr = cnns[:7]
  Y1_tr = np.array([cnn_syn_func1(x) for x in X1_tr])
  X1_te = cnns[7:]
  Y1_te = np.array([cnn_syn_func1(x) for x in X1_te])
  nn_type_1 = 'cnn'
  # MLP regression
  regmlps = generate_mlp_architectures('reg')
  np.random.shuffle(regmlps)
  X2_tr = regmlps[:5]
  Y2_tr = np.array([mlp_syn_func1(x) for x in X2_tr])
  X2_te = regmlps[5:]
  Y2_te = np.array([mlp_syn_func1(x) for x in X2_te])
  nn_type_2 = 'mlp-reg'
  # MLP classification
  classmlps = generate_mlp_architectures('class')
  np.random.shuffle(classmlps)
  X3_tr = classmlps[:5]
  Y3_tr = np.array([syn_func1_common(x) for x in X3_tr])
  X3_te = classmlps[5:]
  Y3_te = np.array([syn_func1_common(x) for x in X3_te])
  nn_type_3 = 'mlp-class'
  return [(X1_tr, Y1_tr, X1_te, Y1_te, nn_type_1),
          (X2_tr, Y2_tr, X2_te, Y2_te, nn_type_2),
          (X3_tr, Y3_tr, X3_te, Y3_te, nn_type_3)]

# Some utilities we will need for testing ------------------------------------------------
def build_nngp_with_dataset(dataset, kernel_type, num_coeffs, dist_type):
  """ Builds a GP using the training set in dataset. """
  mean_func = lambda x: np.array([np.median(dataset[1])] * len(x))
  noise_var = (dataset[1].std() ** 2)/20
  kernel_hyperparams = get_kernel_hyperparams(num_coeffs, kernel_type, dist_type)
  return nn_gp.NNGP(dataset[0], dataset[1], kernel_type, dataset[-1], kernel_hyperparams,
                    mean_func, noise_var, dist_type)

def get_kernel_hyperparams(num_coeffs, kernel_type, dist_type):
  """ Returns the kernel hyperparams for the unit-tests below. """
  # data
  mislabel_coeffs = [2.0, 2.0, 1.0, 1.0, 1.0]
  struct_coeffs = [0.25, 0.5, 1.0, 2.0, 4.0]
  # kernel params
  mislabel_coeffs = mislabel_coeffs[:num_coeffs]
  struct_coeffs = struct_coeffs[:num_coeffs]
  lp_betas = [1e-6] * num_coeffs
  emd_betas = [1] * num_coeffs
  lp_powers = [1] * num_coeffs
  emd_powers = [2] * num_coeffs
  if dist_type == 'lp':
    betas = lp_betas
    powers = lp_powers
  elif dist_type == 'emd':
    betas = emd_betas
    powers = emd_powers
  elif dist_type == 'lp-emd':
    betas = [j for i in zip(lp_betas, emd_betas) for j in i]
    powers = [j for i in zip(lp_powers, emd_powers) for j in i]
  else:
    raise ValueError('Unknown dist_type: %s.'%(dist_type))
  # Now construct the dictionary
  kernel_hyperparams = {}
  kernel_hyperparams['mislabel_coeffs'] = mislabel_coeffs
  kernel_hyperparams['struct_coeffs'] = struct_coeffs
  kernel_hyperparams['betas'] = betas
  kernel_hyperparams['powers'] = powers
  kernel_hyperparams['non_assignment_penalty'] = 1.0
  if kernel_type in ['lpemd_prod', 'lp', 'emd']:
    kernel_hyperparams['scale'] = 1.0
  elif kernel_type in ['lpemd_sum']:
    kernel_hyperparams['alphas'] = [1.0, 1.0]
  return kernel_hyperparams

def fit_nngp_with_dataset(dataset, kernel_type, dist_type):
  """ Fits an NNGP to this dataset. """
  options = load_options(nn_gp.nn_gp_args, '')
  options.kernel_type = kernel_type
  options.dist_type = dist_type
  gp_fitter = nn_gp.NNGPFitter(dataset[0], dataset[1], dataset[-1],
                               options=options, reporter=None)
  _, fitted_gp, _ = gp_fitter.fit_gp()
  return fitted_gp

def compute_average_prediction_error(dataset, preds, true_labels_idx=None):
  """ Computes the prediction error. """
  true_labels_idx = 3 if true_labels_idx is None else true_labels_idx
  return (np.linalg.norm(dataset[true_labels_idx] - preds)**2)/ \
          len(dataset[true_labels_idx])

# Test classes =========================================================================
class NNGPTestCase(BaseTestClass):
  """ Contains unit tests for the TransportNNDistanceComputer class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNGPTestCase, self).__init__(*args, **kwargs)
    self.datasets = gen_gp_test_data()
    self.dist_types = ['lp', 'emd', 'lp-emd']
    self.kernel_types = {'lp': ['lp'], 'emd':['emd'],
                         'lp-emd':['lpemd_prod', 'lpemd_sum']}

  def test_basics(self):
    """ Tests for adding data, evaluation and marginal likelihood. """
    self.report('Testing adding data, evaluation and marginal likelihood.' +
                ' Probabilistic test, might fail.')
    num_coeffs_vals = [2, 1, 4, 5] * 5
    num_tests = 0
    num_successes = 0
    for dataset in self.datasets:
      for dist_type in self.dist_types:
        for kernel_type in self.kernel_types[dist_type]:
          curr_num_coeffs = num_coeffs_vals.pop(0)
          curr_gp = build_nngp_with_dataset(dataset, kernel_type, curr_num_coeffs,
                                            dist_type)
          # Predictions & Marginal likelihood
          curr_preds, _ = curr_gp.eval(dataset[2], 'std')
          curr_gp_err = compute_average_prediction_error(dataset, curr_preds)
          const_err = compute_average_prediction_error(dataset, dataset[1].mean())
          lml = curr_gp.compute_log_marginal_likelihood()
          is_success = curr_gp_err < const_err
          num_tests += 1
          num_successes += is_success
          self.report(('(%s, ntr=%d, nte=%d):: GP-lml=%0.4f, GP-err=%0.4f, ' +
                       'Const-err=%0.4f.  succ=%d')%(dataset[-1][:5], len(dataset[0]),
                       len(dataset[2]), lml, curr_gp_err, const_err, is_success),
                       'test_result')
    succ_frac = num_successes / float(num_tests)
    self.report('Summary: num_successes / num_floats = %d/%d = %0.4f'%(num_successes,
                num_tests, succ_frac), 'test_result')
    assert succ_frac > 0.5

  def test_hallucinated_predictions(self):
    """ Testing hallucinated predictions for NNGP. """
    self.report('Testing hallucinated predictions for NNGP.')
    num_coeffs_vals = [2, 1, 4, 5] * 5
    for dataset in self.datasets:
      for dist_type in self.dist_types:
        for kernel_type in self.kernel_types[dist_type]:
          curr_num_coeffs = num_coeffs_vals.pop(0)
          curr_gp = build_nngp_with_dataset(dataset, kernel_type, curr_num_coeffs,
                                            dist_type)
          curr_preds, curr_stds = curr_gp.eval(dataset[2][1:], 'std')
          ha_preds, ha_stds = curr_gp.eval_with_hallucinated_observations(
                                dataset[2][1:], dataset[0][4:] + dataset[2][:1], 'std')
          assert np.linalg.norm(curr_preds - ha_preds) < _TOL
          assert np.all(curr_stds > ha_stds)

def return_kernel_hyperparams_as_str(kernel_hyperparams):
  """ Returns the kernel hyper-params as a string. """
  if 'scale' in kernel_hyperparams:
    scale_str = 'scale=%0.5f'%(kernel_hyperparams['scale'])
  else:
    scale_str = 'alphas=%s'%(get_list_of_floats_as_str(kernel_hyperparams['alphas'], 5))
  betas_str = 'betas=' + str(kernel_hyperparams['betas'])
  return scale_str + ', ' + betas_str


class NNGPFitterTestCase(BaseTestClass):
  """ Contains unit tests for the TransportNNDistanceComputer class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNGPFitterTestCase, self).__init__(*args, **kwargs)
    self.datasets = gen_gp_test_data()
    self.dist_types = ['lp', 'emd', 'lp-emd']
    self.kernel_types = {'lp': ['lp'], 'emd':['emd'],
                         'lp-emd':['lpemd_prod', 'lpemd_sum']}

  @classmethod
  def _read_kernel_hyperparams(cls, kernel_hyperparams):
    """ Returns the kernel hyper-params as a string. """
    return return_kernel_hyperparams_as_str(kernel_hyperparams)

  def test_marg_likelihood_and_prediction(self):
    """ Tests marginal likelihood and prediction. """
    # pylint: disable=too-many-locals
    self.report('Testing evaluation and marginal likelihood on Fitted GP.' +
                ' Probabilistic test, might fail.')
    naive_num_coeffs_vals = [2, 1, 4, 5] * 5
    num_tests = 0
    num_lml_successes = 0
    num_const_err_successes = 0
    num_naive_err_successes = 0
    for dataset_idx, dataset in enumerate(self.datasets):
      for dist_type in self.dist_types:
        for kernel_type in self.kernel_types[dist_type]:
          # Build the naive GP
          curr_num_coeffs = naive_num_coeffs_vals.pop(0)
          naive_gp = build_nngp_with_dataset(dataset, kernel_type, curr_num_coeffs,
                                            dist_type)
          # Obtain a fitted GP
          fitted_gp = fit_nngp_with_dataset(dataset, kernel_type, dist_type)
          # Obtain log marginal likelihoods
          naive_lml = naive_gp.compute_log_marginal_likelihood()
          fitted_lml = fitted_gp.compute_log_marginal_likelihood()
          lml_succ = naive_lml < fitted_lml
          num_lml_successes += lml_succ
          self.report(('(%s, ntr=%d, nte=%d):: naive-gp-lml=%0.4f, ' +
                       'fitted-gp-lml=%0.4f, lml-succ=%d.')%(
                       dataset[-1][:5], len(dataset[0]), len(dataset[2]), naive_lml,
                       fitted_lml, lml_succ), 'test_result')
          # Predictions & Marginal likelihood
          naive_preds, _ = naive_gp.eval(dataset[2], 'std')
          naive_gp_err = compute_average_prediction_error(dataset, naive_preds)
          fitted_preds, _ = fitted_gp.eval(dataset[2], 'std')
          fitted_gp_err = compute_average_prediction_error(dataset, fitted_preds)
          const_err = compute_average_prediction_error(dataset, dataset[1].mean())
          const_err_succ = const_err > fitted_gp_err
          naive_err_succ = naive_gp_err > fitted_gp_err
          num_const_err_successes += const_err_succ
          num_naive_err_successes += naive_err_succ
          self.report(('  dataset #%d: const-err: %0.4f (%d), naive-gp-err=%0.4f (%d), ' +
                       'fitted-gp-err=%0.4f.')%(
                       dataset_idx, const_err, const_err_succ,
                       naive_gp_err, naive_err_succ, fitted_gp_err),
                      'test_result')
          # Print out betas
          self.report('  fitted kernel %s hyper-params: %s'%(kernel_type,
                      self._read_kernel_hyperparams(fitted_gp.kernel.hyperparams)),
                      'test_result')
          num_tests += 1
    # Print out some statistics
    lml_frac = float(num_lml_successes) / float(num_tests)
    const_err_frac = float(num_const_err_successes) / float(num_tests)
    naive_err_frac = float(num_naive_err_successes) / float(num_tests)
    self.report('LML tests success fraction = %d/%d = %0.4f'%(
                num_lml_successes, num_tests, lml_frac), 'test_result')
    self.report('Const-err success fraction = %d/%d = %0.4f'%(
                num_const_err_successes, num_tests, const_err_frac), 'test_result')
    self.report('Naive-GP-err success fraction = %d/%d = %0.4f'%(
                num_naive_err_successes, num_tests, naive_err_frac), 'test_result')
    assert num_lml_successes == num_tests
    assert const_err_frac >= 0.5
    assert naive_err_frac >= 0.3


if __name__ == '__main__':
  execute_tests()

