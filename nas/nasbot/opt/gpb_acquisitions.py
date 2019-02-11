"""
  Acquisition functions for Bayesian Optimisation.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class
# pylint: disable=no-name-in-module
# pylint: disable=star-args

from argparse import Namespace
import numpy as np
from scipy.stats import norm as normal_distro

def _optimise_acquisition(acq_fn, acq_optimiser, anc_data):
  """ All methods will just call this. """
  if anc_data.acq_opt_method == 'direct':
    acquisition = lambda x: acq_fn(x.reshape((1, -1)))
  else:
    acquisition = acq_fn
  _, opt_pt = acq_optimiser(acquisition, anc_data.max_evals)
  return opt_pt

def _get_halluc_points(_, halluc_pts):
  """ Re-formats halluc_pts if necessary. """
  if len(halluc_pts) > 0:
    return halluc_pts
  else:
    return halluc_pts

# Thompson sampling ---------------------------------------------------------------
def asy_ts(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via TS in the asyuential setting. """
  gp_sample = lambda x: gp.draw_samples(1, X_test=x, mean_vals=None, covar=None).ravel()
  return _optimise_acquisition(gp_sample, acq_optimiser, anc_data)

def asy_hts(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via TS using hallucinated observaitons in the asynchronus
      setting. """
  halluc_pts = _get_halluc_points(gp, anc_data.evals_in_progress)
  gp_sample = lambda x: gp.draw_samples_with_hallucinated_observations(1, x,
                                                                       halluc_pts).ravel()
  return _optimise_acquisition(gp_sample, acq_optimiser, anc_data)

def syn_ts(num_workers, gp, acq_optimiser, anc_data, **kwargs):
  """ Returns a batch of recommendations via TS in the synchronous setting. """
  recommendations = []
  for _ in range(num_workers):
    rec_j = asy_ts(gp, acq_optimiser, anc_data, **kwargs)
    recommendations.append(rec_j)
  return recommendations

# UCB ------------------------------------------------------------------------------
def _get_gp_ucb_dim(gp):
  """ Returns the dimensionality of the dimension. """
  if hasattr(gp, 'ucb_dim') and gp.ucb_dim is not None:
    return gp.ucb_dim
  elif hasattr(gp.kernel, 'dim'):
    return gp.kernel.dim
  else:
    return 3.0

def _get_ucb_beta_th(dim, time_step):
  """ Computes the beta t for UCB based methods. """
  return np.sqrt(0.5 * dim * np.log(2 * dim * time_step + 1))

def asy_ucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB in the asyuential setting. """
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  def _ucb_acq(x):
    """ Computes the GP-UCB acquisition. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu + beta_th * sigma
  return _optimise_acquisition(_ucb_acq, acq_optimiser, anc_data)

def _halluc_ucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB using hallucinated inputs in the asynchronous
      setting. """
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  halluc_pts = _get_halluc_points(gp, anc_data.evals_in_progress)
  def _ucb_halluc_acq(x):
    """ Computes GP-UCB acquisition with hallucinated observations. """
    mu, sigma = gp.eval_with_hallucinated_observations(x, halluc_pts, uncert_form='std')
    return mu + beta_th * sigma
  return _optimise_acquisition(_ucb_halluc_acq, acq_optimiser, anc_data)

def asy_hucb(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB using hallucinated inputs in the asynchronous
      setting. """
  return _halluc_ucb(gp, acq_optimiser, anc_data)

def syn_hucb(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via Batch UCB in the synchronous setting. """
  recommendations = [asy_ucb(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    anc_data.evals_in_progress = recommendations
    recommendations.append(_halluc_ucb(gp, acq_optimiser, anc_data))
  return recommendations

def syn_ucbpe(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via UCB-PE in the synchronous setting. """
  # Define some internal functions.
  beta_th = _get_ucb_beta_th(_get_gp_ucb_dim(gp), anc_data.t)
  # 1. An LCB for the function
  def _ucbpe_lcb(x):
    """ An LCB for GP-UCB-PE. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu - beta_th * sigma
  # 2. A modified UCB for the function using hallucinated observations
  def _ucbpe_2ucb(x):
    """ A UCB for GP-UCB-PE. """
    mu, sigma = gp.eval(x, uncert_form='std')
    return mu + 2 * beta_th * sigma
  # 3. UCB-PE acquisition for the 2nd point in the batch and so on.
  def _ucbpe_acq(x, yt_dot, halluc_pts):
    """ Acquisition for GP-UCB-PE. """
    _, halluc_stds = gp.eval_with_hallucinated_observations(x, halluc_pts,
                                                            uncert_form='std')
    return (_ucbpe_2ucb(x) > yt_dot).astype(np.double) * halluc_stds

  # Now the algorithm
  yt_dot_arg = _optimise_acquisition(_ucbpe_lcb, acq_optimiser, anc_data)
  yt_dot = _ucbpe_lcb(yt_dot_arg.reshape((-1, _get_gp_ucb_dim(gp))))
  recommendations = [asy_ucb(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    curr_acq = lambda x: _ucbpe_acq(x, yt_dot, np.array(recommendations))
    new_rec = _optimise_acquisition(curr_acq, acq_optimiser, anc_data)
    recommendations.append(new_rec)
  return recommendations

# EI stuff ----------------------------------------------------------------------------
def asy_ei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation based on GP-EI. """
  curr_best = anc_data.curr_max_val
  def _ei_acq(x):
    """ Acquisition for GP EI. """
    mu, sigma = gp.eval(x, uncert_form='std')
    Z = (mu - curr_best) / sigma
    return (mu - curr_best)*normal_distro.cdf(Z) + sigma*normal_distro.pdf(Z)
  return _optimise_acquisition(_ei_acq, acq_optimiser, anc_data)

def _halluc_ei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation based on GP-HEI using hallucinated points. """
  halluc_pts = _get_halluc_points(gp, anc_data.evals_in_progress)
  curr_best = anc_data.curr_max_val
  def _ei_halluc_acq(x):
    """ Computes the hallucinated EI acquisition. """
    mu, sigma = gp.eval_with_hallucinated_observations(x, halluc_pts, uncert_form='std')
    Z = (mu - curr_best) / sigma
    return (mu - curr_best)*normal_distro.cdf(Z) + sigma*normal_distro.pdf(Z)
  return _optimise_acquisition(_ei_halluc_acq, acq_optimiser, anc_data)

def asy_hei(gp, acq_optimiser, anc_data):
  """ Returns a recommendation via EI using hallucinated inputs in the asynchronous
      setting. """
  return _halluc_ei(gp, acq_optimiser, anc_data)

def syn_hei(num_workers, gp, acq_optimiser, anc_data):
  """ Returns a recommendation via EI in the synchronous setting. """
  recommendations = [asy_ei(gp, acq_optimiser, anc_data)]
  for _ in range(1, num_workers):
    anc_data.evals_in_progress = recommendations
    recommendations.append(_halluc_ei(gp, acq_optimiser, anc_data))
  return recommendations

# Random --------------------------------------------------------------------------------
def asy_rand(_, acq_optimiser, anc_data):
  """ Returns random values for the acquisition. """
  def _rand_eval(_):
    """ Acquisition for asy_rand. """
    return np.random.random((1,))
  return _optimise_acquisition(_rand_eval, acq_optimiser, anc_data)

def syn_rand(num_workers, gp, acq_optimiser, anc_data):
  """ Returns random values for the acquisition. """
  return [asy_rand(gp, acq_optimiser, anc_data) for _ in range(num_workers)]


# Put all of them into the following namespaces.
syn = Namespace(
  # UCB
  hucb=syn_hucb,
  ucbpe=syn_ucbpe,
  # TS
  ts=syn_ts,
  # EI
  hei=syn_hei,
  # Rand
  rand=syn_rand,
  )

asy = Namespace(
  # UCB
  ucb=asy_ucb,
  hucb=asy_hucb,
  # EI
  ei=asy_ei,
  hei=asy_hei,
  # TS
  ts=asy_ts,
  hts=asy_hts,
  # Rand
  rand=asy_rand,
  )

seq = Namespace(
  ucb=asy_ucb,
  ts=asy_ts,
  )

