"""
  Unit tests for gp_instances.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

# Local
from nas.nasbot.gp.gp_instances import SimpleGPFitter
from nas.nasbot.utils.base_test_class import BaseTestClass, execute_tests
from nas.nasbot.gp.unittest_gp_core import build_gp_with_dataset, compute_average_prediction_error
from nas.nasbot.gp.unittest_gp_core import gen_gp_test_data


def fit_gp_with_dataset(dataset):
  """ A wrapper to fit a gp using the dataset. """
  _, fitted_gp, _ = (SimpleGPFitter(dataset[0], dataset[1], reporter=None)).fit_gp()
  return fitted_gp


class SimpleGPFitterTestCase(BaseTestClass):
  """ Unit tests for the GP class. """
  # pylint: disable=too-many-locals

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(SimpleGPFitterTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Set up for the tests. """
    self.datasets = gen_gp_test_data()

  def test_marg_likelihood(self):
    """ This tests for the marginal likelihood. Since we are fitting the hyper parameters
        by maximising the marginal likelihood it should have a higher value. """
    self.report('Marginal likelihood. Probabilistic test, might fail.')
    num_successes = 0
    for dataset in self.datasets:
      naive_gp = build_gp_with_dataset(dataset)
      fitted_gp = fit_gp_with_dataset(dataset)
      naive_lml = naive_gp.compute_log_marginal_likelihood()
      fitted_lml = fitted_gp.compute_log_marginal_likelihood()
      success = naive_lml <= fitted_lml
      self.report('(N,D)=%s:: naive-lml=%0.4f, fitted-lml=%0.4f, succ=%d'%(
          str(dataset[0].shape), naive_lml, fitted_lml, success), 'test_result')
      num_successes += success
    assert num_successes == len(self.datasets)

  def test_prediction(self):
    """ Tests for prediction on a test set.
    """
    self.report('Prediction. Probabilistic test, might fail.')
    num_successes = 0
    for dataset in self.datasets:
      naive_gp = build_gp_with_dataset(dataset)
      naive_preds, _ = naive_gp.eval(dataset[3])
      naive_err = compute_average_prediction_error(dataset, naive_preds)
      fitted_gp = fit_gp_with_dataset(dataset)
      fitted_preds, _ = fitted_gp.eval(dataset[3])
      fitted_err = compute_average_prediction_error(dataset, fitted_preds)
      success = fitted_err <= naive_err
      self.report('(N,D)=%s:: naive-err=%0.4f, fitted-err=%0.4f, succ=%d'%(
          str(dataset[0].shape), naive_err, fitted_err, success), 'test_result')
      self.report('  -- GP: %s'%(str(fitted_gp)), 'test_result')
      num_successes += success
    assert num_successes > 0.6 *len(self.datasets)


if __name__ == '__main__':
  execute_tests(5424)

