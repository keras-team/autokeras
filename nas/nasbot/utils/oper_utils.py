"""
  A collection of operational utilities we will need.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-name-in-module
# pylint: disable=invalid-name

import numpy as np
import ot
from warnings import warn
# Local imports
try:
  import utils.direct_fortran.direct as direct_ft_wrap
except ImportError:
  direct_ft_wrap = None
from ..utils.general_utils import map_to_bounds

_MAX_DIRECT_FN_EVALS = 2.6e6 # otherwise the fortran software complains

# Optimal transport and Earth mover's distance ===========================================
def opt_transport(supply, demand, costs):
  """ A wrapper for the EMD computation using the Optimal Transport (ot) package.
      if emd_only is False, it only returns the emd value. Else it returns the transport
      matrix and the minimum value of the objective.
  """
  supply = supply.astype(np.float64)
  demand = demand.astype(np.float64)
  tot_supply = supply.sum()
  tot_demand = demand.sum()
#   assert tot_supply == tot_demand
  supply = supply / tot_supply
  demand = demand / tot_demand
  # Now solve the problem
  T = ot.emd(supply, demand, costs)
  T = tot_supply * T
  min_val = np.sum(T * costs)
  emd = min_val/tot_supply
  return T, min_val, emd

# Various utilities for global optimisation of *cheap* functions =========================
# Random samplning
def random_sample(obj, bounds, max_evals, vectorised=True):
  """ Optimises a function by randomly sampling and choosing its maximum. """
  dim = len(bounds)
  rand_pts = map_to_bounds(np.random.random((int(max_evals), dim)), bounds)
  if vectorised:
    obj_vals = obj(rand_pts)
  else:
    obj_vals = np.array([obj(x) for x in rand_pts])
  return rand_pts, obj_vals

# Random maximisation
def random_maximise(obj, bounds, max_evals, vectorised=True):
  """ Optimises a function by randomly sampling and choosing its maximum. """
  rand_pts, obj_vals = random_sample(obj, bounds, max_evals, vectorised)
  max_idx = obj_vals.argmax()
  max_val = obj_vals[max_idx]
  max_pt = rand_pts[max_idx]
  return max_val, max_pt

# Maximisation with DIviding RECTangles (direct) -----------------------------------------
def direct_ft_minimise(obj, lower_bounds, upper_bounds, max_evals,
                       eps=1e-5,
                       return_history=False,
                       max_iterations=None,
                       alg_method=0,
                       fglobal=-1e100,
                       fglper=0.01,
                       volper=-1.0,
                       sigmaper=-1.0,
                       log_file_name='',
                       vectorised=False,
                       alternative_if_direct_not_loaded='rand',
                      ):
  """
    A wrapper for the fortran implementation. The four mandatory arguments are self
    explanatory. If return_history is True it also returns the history of evaluations.
    max_iterations is the maximum number of iterations of the direct algorithm.
    I am not sure what the remaining arguments are for.
  """
  # pylint: disable=too-many-locals
  # pylint: disable=too-many-arguments
  # If you want an implementation that reads the history off the log file as well, you
  # can find one in the Thompson directory. I modified the original fortran code
  # to write and then read the history of queried values. --kirthevasan

  if direct_ft_wrap is None:
    report_str = 'Attempted to use direct, but fortran library could not be imported.'
    if alternative_if_direct_not_loaded is None:
      report_str += ' Alternative not specified. Raising exception.'
      raise Exception(report_str)
    elif alternative_if_direct_not_loaded.lower().startswith('rand'):
      report_str += 'Using random optimiser.'
      warn(report_str)
      bounds = np.array([lower_bounds, upper_bounds]).T
      max_val, max_pt = random_maximise(obj, bounds, max_evals, vectorised)
      return max_val, max_pt, None
    else:
      report_str += 'Unknown option for alternative_if_direct_not_loaded: %s'%(
                     alternative_if_direct_not_loaded)
      raise ValueError(report_str)

  # --------------
  # Preliminaries.
  max_evals = min(_MAX_DIRECT_FN_EVALS, max_evals) # otherwise the fortran sw complains.
  max_iterations = max_evals if max_iterations is None else max_iterations
  lower_bounds = np.array(lower_bounds, dtype=np.float64)
  upper_bounds = np.array(upper_bounds, dtype=np.float64)
  if len(lower_bounds) != len(upper_bounds):
    raise ValueError('The dimensionality of the lower and upper bounds should match.')

  # Create a wrapper to comply with the fortran requirements.
  def _objective_wrap(x, *_):
    """ A wrapper to comply with the fortran requirements. """
    return (obj(x), 0)

  # Some dummy data to comply with the fortran requirements.
  iidata = np.ones(0, dtype=np.int32)
  ddata = np.ones(0, dtype=np.float64)
  cdata = np.ones([0, 40], dtype=np.uint8)
  # Call the function.
  min_pt, min_val, _ = direct_ft_wrap.direct(_objective_wrap,
                                             eps,
                                             max_evals,
                                             max_iterations,
                                             lower_bounds,
                                             upper_bounds,
                                             alg_method,
                                             log_file_name,
                                             fglobal,
                                             fglper,
                                             volper,
                                             sigmaper,
                                             iidata,
                                             ddata,
                                             cdata
                                            )
  if return_history:
    # TODO: implement this. Read it off the log file. There is a
    # If you want an implementation that reads the history off the log file as well, you
    # can find one in the Thompson directory. --Samy
    pass
  else:
    history = None
  # return
  return min_val, min_pt, history


def direct_ft_maximise(obj, lower_bounds, upper_bounds, max_evals, **kwargs):
  """
    A wrapper for maximising a function which calls direct_ft_minimise. See arguments
    under direct_ft_minimise for more details.
  """
  min_obj = lambda x: -obj(x)
  min_val, max_pt, history = direct_ft_minimise(min_obj, lower_bounds, upper_bounds,
                                                max_evals, **kwargs)
  max_val = - min_val
  # TODO: Fix history here.
  return max_val, max_pt, history

