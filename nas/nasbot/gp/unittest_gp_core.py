"""
  Unit tests for gp.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

from nas.nasbot.utils.base_test_class import BaseTestClass, execute_tests
import numpy as np
# Local
from nas.nasbot.gp import gp_core
from nas.nasbot.gp import kernel


def gen_gp_test_data():
  """ This function generates a bunch of test data. """
  # pylint: disable=too-many-locals
  # Dataset 1
  f1 = lambda x: (x**2).sum(axis=1)
  N1 = 5
  X1_tr = np.array(range(N1)).astype(float).reshape((N1, 1))/N1 + 1/(2*N1)
  Y1_tr = f1(X1_tr)
  X1_te = np.random.random((50, 1))
  Y1_te = f1(X1_te)
  kernel1 = kernel.SEKernel(1, 1, 0.5)
  # Dataset 2
  N2 = 100
  D2 = 10
  f2 = lambda x: ((x**2) * range(1, D2+1)/D2).sum(axis=1)
  X2_tr = np.random.random((N2, D2))
  Y2_tr = f2(X2_tr)
  X2_te = np.random.random((N2, D2))
  Y2_te = f2(X2_te)
  kernel2 = kernel.SEKernel(D2, 10, 0.2*np.sqrt(D2))
  # Dataset 3
  N3 = 200
  D3 = 6
  f3 = lambda x: ((x**3 + 2 * x**2 - x + 2) * range(1, D3+1)/D3).sum(axis=1)
  X3_tr = np.random.random((N3, D3))
  Y3_tr = f3(X3_tr)
  X3_te = np.random.random((N3, D3))
  Y3_te = f3(X3_te)
  kernel3 = kernel.SEKernel(D3, 10, 0.2*np.sqrt(D3))
  # Dataset 4
  N4 = 400
  D4 = 8
  f4 = lambda x: ((np.sin(x**2) + 2 * np.cos(x**2) - x + 2) *
                  range(1, D4+1)/D4).sum(axis=1)
  X4_tr = np.random.random((N4, D4))
  Y4_tr = f4(X4_tr)
  X4_te = np.random.random((N4, D4))
  Y4_te = f4(X4_te)
  kernel4 = kernel.SEKernel(D4, 10, 0.2*np.sqrt(D4))
  # put all datasets into a list.
  return [(X1_tr, Y1_tr, kernel1, X1_te, Y1_te),
          (X2_tr, Y2_tr, kernel2, X2_te, Y2_te),
          (X3_tr, Y3_tr, kernel3, X3_te, Y3_te),
          (X4_tr, Y4_tr, kernel4, X4_te, Y4_te)]

def build_gp_with_dataset(dataset):
  """ Internal function to build a GP with the dataset. """
  mean_func = lambda x: np.array([np.median(dataset[1])] * len(x))
  noise_var = dataset[1].std()**2/20
  return gp_core.GP(dataset[0], dataset[1], dataset[2], mean_func, noise_var)

def compute_average_prediction_error(dataset, preds):
  """ Computes the prediction error. """
  return (np.linalg.norm(dataset[4] - preds)**2)/len(dataset[4])


class GPTestCase(BaseTestClass):
  """ Unit tests for the GP class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(GPTestCase, self).__init__(*args, **kwargs)

  def setUp(self):
    """ Set up for tests. """
    self.datasets = gen_gp_test_data()

  def test_add_data(self):
    """ Tests GP.add_data. """
    self.report('GP.add_data')
    for dataset in self.datasets:
      num_new = np.random.randint(3, 10)
      X_new = np.random.random((num_new, dataset[0].shape[1]))
      Y_new = np.random.random(num_new)
      curr_gp = build_gp_with_dataset(dataset)
      curr_gp.add_data(X_new, Y_new)
      assert num_new + len(dataset[1]) == curr_gp.num_tr_data

  def test_eval(self):
    """ Tests the evaluation. """
    self.report('GP.eval: Probabilistic test, might fail sometimes')
    num_successes = 0
    for dataset in self.datasets:
      curr_gp = build_gp_with_dataset(dataset)
      curr_pred, _ = curr_gp.eval(dataset[3])
      curr_err = compute_average_prediction_error(dataset, curr_pred)
      const_err = compute_average_prediction_error(dataset, dataset[1].mean())
      success = curr_err < const_err
      self.report(('(N,D)=' + str(dataset[0].shape) + ':: GP-err= ' + str(curr_err) +
                   ',   Const-err= ' + str(const_err) + ',  success=' + str(success)),
                  'test_result')
      num_successes += int(success)
    assert num_successes > 0.6 *len(self.datasets)

  def test_compute_log_marginal_likelihood(self):
    """ Tests compute_log_marginal_likelihood. Does not test for accurate implementation.
        Only tests if the function runs without runtime errors. """
    self.report('GP.compute_log_marginal_likelihood: ** Runtime test errors only **')
    for dataset in self.datasets:
      curr_gp = build_gp_with_dataset(dataset)
      lml = curr_gp.compute_log_marginal_likelihood()
      self.report('(N,D)=' + str(dataset[0].shape) + ' lml = ' + str(lml),
                  'test_result')

  def test_hallucinated_predictions(self):
    """ Tests predictions with hallucinated observations. """
    self.report('GP.eval_with_hallucinated_observations.')
    for dataset in self.datasets:
      curr_gp = build_gp_with_dataset(dataset)
      pred, stds = curr_gp.eval(dataset[3], 'std')
      # Add hallucinations
      X_halluc = np.random.random((max(dataset[0].shape[0]/10, 5), dataset[0].shape[1]))
      ha_pred, ha_stds = curr_gp.eval_with_hallucinated_observations(dataset[3], X_halluc,
                                  'std')
      assert np.linalg.norm(pred - ha_pred) <= 1e-5
      assert np.all(stds > ha_stds)


if __name__ == '__main__':
  execute_tests()

