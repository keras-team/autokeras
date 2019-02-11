"""
  A collection of utilities for MF-GP Bandits.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import
# pylint: disable=super-on-old-class

import numpy as np
# Local imports
from ..opt.function_caller import get_euc_function_caller_from_function

# Hartmann Functions ---------------------------------------------------------------------
def hartmann(x, alpha, A, P, max_val=np.inf):
  """ Computes the hartmann function for any given A and P. """
  log_sum_terms = (A * (P - x)**2).sum(axis=1)
  return min(max_val, alpha.dot(np.exp(-log_sum_terms)))

def _get_hartmann_data(domain_dim):
  """ Returns A and P for the 3D hartmann function. """
  # pylint: disable=bad-whitespace
  if domain_dim == 3:
    A = np.array([[3.0, 10, 30],
                  [0.1, 10, 35],
                  [3.0, 10, 30],
                  [0.1, 10, 35]], dtype=np.float64)
    P = 1e-4 * np.array([[3689, 1170, 2673],
                         [4699, 4387, 7470],
                         [1091, 8732, 5547],
                         [ 381, 5743, 8828]], dtype=np.float64)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    domain = [[0, 1]] * 3
    opt_pt = np.array([0.114614, 0.555649, 0.852547])
    max_val = 3.86278

  elif domain_dim == 6:
    A = np.array([[  10,   3,   17, 3.5, 1.7,  8],
                  [0.05,  10,   17, 0.1,   8, 14],
                  [   3, 3.5,  1.7,  10,  17,  8],
                  [  17,   8, 0.05,  10, 0.1, 14]], dtype=np.float64)
    P = 1e-4 * np.array([[1312, 1696, 5569,  124, 8283, 5886],
                         [2329, 4135, 8307, 3736, 1004, 9991],
                         [2348, 1451, 3522, 2883, 3047, 6650],
                         [4047, 8828, 8732, 5743, 1091,  381]], dtype=np.float64)
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    domain = [[0, 1]] * 6
    opt_pt = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    max_val = 3.322368

  else:
    raise NotImplementedError('Only implemented in 3 and 6 dimensions.')
  return A, P, alpha, opt_pt, domain, max_val

def get_hartmann_high_d_function_caller(domain_dim, **kwargs):
  """ Constructs a higher dimensional Hartmann function. """
  group_dim = 6
  num_groups = int(domain_dim/group_dim)
  A, P, alpha, l_opt_pt, l_domain_bounds, l_max_val = _get_hartmann_data(group_dim)
  hartmann_func = lambda x: hartmann(x, alpha, A, P, l_max_val)
  def _eval_highd_hartmann_func(x):
    """ Evaluates the higher dimensional hartmann function. """
    ret = 0
    for j in range(num_groups):
      ret += hartmann_func(x[j*group_dim:(j+1)*group_dim])
    return ret
  opt_pt = np.tile(l_opt_pt, num_groups+1)[0:domain_dim]
  opt_val = num_groups * l_max_val
  domain_bounds = np.tile(np.array(l_domain_bounds).T, num_groups+1).T[0:domain_dim]
  return get_euc_function_caller_from_function(_eval_highd_hartmann_func, domain_bounds,
           vectorised=False, opt_pt=opt_pt, opt_val=opt_val, **kwargs)

def get_hartmann_high_d_function_caller_from_descr(descr, **kwargs):
  """ Constructs a high dimensional hartmann function from a string. """
  domain_dim = int(descr[8:])
  return get_hartmann_high_d_function_caller(domain_dim, **kwargs)

def get_hartmann_function_caller(domain_dim, **kwargs):
  """ Returns a FunctionCaller object for the hartmann function. """
  A, P, alpha, opt_pt, domain_bounds, max_val = _get_hartmann_data(domain_dim)
  hartmann_func = lambda x: hartmann(x, alpha, A, P, max_val)
  return get_euc_function_caller_from_function(hartmann_func, domain_bounds,
                                               vectorised=False,
                                               opt_pt=opt_pt, opt_val=max_val, **kwargs)
# Hartmann Functions end here ------------------------------------------------------------

# Shekel Function ------------------------------------------------------------------------
def shekel(x, C, beta, max_val=np.inf):
  """ Computes the Shekel function for the given C and beta. """
  inv_terms = ((C.T - x)**2).sum(axis=1) + beta
  return min(max_val, (1/inv_terms).sum())

def _get_shekel_data():
  """ Returns the C, beta parameters and optimal values for the shekel function. """
  C = [[4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
       [4, 1, 8, 6, 7, 9, 3, 1, 2, 3],
       [4, 1, 8, 6, 3, 2, 5, 8, 6, 7],
       [4, 1, 8, 6, 7, 9, 3, 1, 2, 3]]
  C = np.array(C, dtype=np.double)
  beta = 0.1 * np.array([1, 2, 2, 4, 4, 6, 3, 7, 5, 5], dtype=np.double)
  opt_pt = np.array([4, 4, 4, 4], dtype=np.double)
  opt_val = shekel(opt_pt, C, beta)
  domain_bounds = [[0, 10]] * 4
  return C, beta, opt_pt, domain_bounds, opt_val

def get_shekel_function_caller(**kwargs):
  """ Returns a FunctionCaller object for the hartmann function. """
  C, beta, opt_pt, domain_bounds, opt_val = _get_shekel_data()
  shekel_func = lambda x: shekel(x, C, beta, opt_val)
  return get_euc_function_caller_from_function(shekel_func, domain_bounds,
           vectorised=False, opt_pt=opt_pt, opt_val=opt_val, **kwargs)

# Shekel function ends here --------------------------------------------------------------


# Currin Exponential Function ------------------------------------------------------------
def currin_exp(x, alpha):
  """ Computes the currin exponential function. """
  x1 = x[0]
  x2 = x[1]
  val_1 = 1 - alpha * np.exp(-1/(2 * x2))
  val_2 = (2300*x1**3 + 1900*x1**2 + 2092*x1 + 60) / (100*x1**3 + 500*x1**2 + 4*x1 + 20)
  return val_1 * val_2

def get_currin_exp_function_caller(**kwargs):
  """ Returns a FunctionCaller object for the Currin Exponential function. """
  currin_exp_func = lambda x: currin_exp(x, 1)
  opt_val = 13.798650
  opt_pt = None
  domain_bounds = np.array([[0, 1], [0, 1]])
  return get_euc_function_caller_from_function(currin_exp_func,
           domain_bounds=domain_bounds, vectorised=False, opt_val=opt_val,
           opt_pt=opt_pt, **kwargs)


# Branin Function ------------------------------------------------------------------------
def branin(x, a, b, c, r, s, t):
  """ Computes the Branin function. """
  x1 = x[0]
  x2 = x[1]
  neg_ret = a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1-t)*np.cos(x1) + s
  return -neg_ret

def _get_branin_data():
  """ Gets the constants for the branin function. """
  a = 1
  b = 5.1/(4*np.pi**2)
  c = 5/np.pi
  r = 6
  s = 10
  t = 1/(8*np.pi)
  opt_pt = np.array([np.pi, 2.275])
  domain_bounds = np.array([[-5, 10], [0, 15]])
  return a, b, c, r, s, t, opt_pt, domain_bounds

def get_branin_function_caller(**kwargs):
  """ Returns a FunctionCaller object for the Branin function. """
  a, b, c, r, s, t, opt_pt, domain_bounds = _get_branin_data()
  branin_func = lambda x: branin(x, a, b, c, r, s, t)
  opt_val = branin_func(opt_pt)
  return get_euc_function_caller_from_function(branin_func, domain_bounds=domain_bounds,
    vectorised=False, opt_val=opt_val, opt_pt=opt_pt, **kwargs)

def get_branin_high_d_function_caller(domain_dim, **kwargs):
  """ Constructs a higher dimensional Hartmann function. """
  group_dim = 2
  num_groups = int(domain_dim/group_dim)
  a, b, c, r, s, t, opt_pt_2d, domain_bounds_2d = _get_branin_data()
  branin_func = lambda x: branin(x, a, b, c, r, s, t)
  def _eval_highd_branin_func(x):
    """ Evaluates higher dimensional branin function. """
    ret = 0
    for j in range(num_groups):
      ret += branin_func(x[j*group_dim:(j+1)*group_dim])
    return ret
  opt_pt = np.tile(opt_pt_2d, num_groups+1)[0:domain_dim]
  opt_val = _eval_highd_branin_func(opt_pt)
  domain_bounds = np.tile(np.array(domain_bounds_2d).T, num_groups+1).T[0:domain_dim]
  return get_euc_function_caller_from_function(_eval_highd_branin_func, domain_bounds,
           vectorised=False, opt_pt=opt_pt, opt_val=opt_val, **kwargs)

def get_branin_high_d_function_caller_from_descr(descr, **kwargs):
  """ Constructs a high dimensional hartmann function from a string. """
  domain_dim = int(descr[7:])
  return get_branin_high_d_function_caller(domain_dim, **kwargs)

# Borehole Function ----------------------------------------------------------------------
def borehole(x, z, max_val):
  """ Computes the Bore Hole function. """
  # pylint: disable=bad-whitespace
  rw = x[0]
  r  = x[1]
  Tu = x[2]
  Hu = x[3]
  Tl = x[4]
  Hl = x[5]
  L  = x[6]
  Kw = x[7]
  # Compute high fidelity function
  frac2 = 2*L*Tu/(np.log(r/rw) * rw**2 * Kw)
  f2 = min(max_val, 2 * np.pi * Tu * (Hu - Hl)/(np.log(r/rw) * (1 + frac2 + Tu/Tl)))
  # Compute low fidelity function
  f1 = 5 * Tu * (Hu - Hl)/(np.log(r/rw) * (1.5 + frac2 + Tu/Tl))
  # Compute final output
  return f2*z + f1*(1-z)

def get_borehole_function_caller(**kwargs):
  """ Returns a FunctionCaller object for the Borehold function. """
  opt_val = 309.523221
  opt_pt = None
  borehole_func = lambda x: borehole(x, 1, opt_val)
  domain_bounds = [[0.05, 0.15],
                   [100, 50000],
                   [63070, 115600],
                   [990, 1110],
                   [63.1, 116],
                   [700, 820],
                   [1120, 1680],
                   [9855, 12045]]
  return get_euc_function_caller_from_function(borehole_func, domain_bounds,
    vectorised=False, opt_val=opt_val, opt_pt=opt_pt, **kwargs)

# Park1 function ==================================================================
def park1(x, max_val=np.inf):
  """ Computes the park1 function. """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret1 = (x1/2) * (np.sqrt(1 + (x2 + x3**2)*x4/(x1**2)) - 1)
  ret2 = (x1 + 3*x4) * np.exp(1 + np.sin(x3))
  return min(ret1 + ret2, max_val)

def get_park1_function_caller(**kwargs):
  """ Returns the park1 function caller. """
  opt_val = 25.5872304
  opt_pt = None
  func = lambda x: park1(x, opt_val)
  domain_bounds = [[0, 1]] * 4
  return get_euc_function_caller_from_function(func, domain_bounds,
           vectorised=False, opt_val=opt_val, opt_pt=opt_pt, **kwargs)

# Park2 function ==================================================================
def park2(x, max_val=np.inf):
  """ Comutes the park2 function """
  x1 = x[0]
  x2 = x[1]
  x3 = x[2]
  x4 = x[3]
  ret = (2.0/3.0) * np.exp(x1 + x2) - x4*np.sin(x3) + x3
  return min(ret, max_val)

def get_park2_function_caller(**kwargs):
  """ Returns function caller for park2. """
  opt_val = 5.925698
  opt_pt = None
  func = lambda x: park2(x, opt_val)
  domain_bounds = [[0, 1]] * 4
  return get_euc_function_caller_from_function(func, domain_bounds,
           vectorised=False, opt_val=opt_val, opt_pt=opt_pt, **kwargs)

def get_high_d_function_from_low_d(domain_dim, group_dim, low_d_func):
  """ Constructs a higher dimensional Hartmann function. """
  num_groups = int(domain_dim/group_dim)
  def _eval_highd_func(x):
    """ Evaluates the higher dimensional function. """
    ret = 0
    for j in range(num_groups):
      ret += low_d_func(x[j*group_dim: (j+1)*group_dim])
    return ret
  return _eval_highd_func, num_groups

def get_high_d_function_caller_from_low_d_func(domain_dim, low_d_func,
      low_d_domain_bounds, low_d_opt_val, low_d_opt_pt, **kwargs):
  """ Gets a low dimensional function caller from a high dimensional one. """
  group_dim = len(low_d_domain_bounds)
  high_d_func, num_groups = get_high_d_function_from_low_d(domain_dim, group_dim,
                                                           low_d_func)
  high_d_domain_bounds = np.tile(np.array(low_d_domain_bounds).T,
                                 num_groups+1).T[0:domain_dim]
  high_d_opt_pt = None
  high_d_opt_val = None
  if low_d_opt_pt is not None:
    high_d_opt_pt = np.tile(low_d_opt_pt, num_groups+1)[0:domain_dim]
    high_d_opt_val = high_d_func(high_d_opt_pt)
  elif low_d_opt_val is not None:
    high_d_opt_val = num_groups * low_d_opt_val
  # Return
  func_caller = get_euc_function_caller_from_function(high_d_func,
                                                  high_d_domain_bounds,
                                                  vectorised=False,
                                                  opt_val=high_d_opt_val,
                                                  opt_pt=high_d_opt_pt,
                                                  **kwargs)
  return func_caller

def get_high_d_function_caller_from_low_d_func_caller(domain_dim,
                                                      low_d_func_caller, **kwargs):
  """ Gets a high dimensional function caller from a low dimensional one. """
  return get_high_d_function_caller_from_low_d_func(domain_dim, low_d_func_caller.func,
           low_d_func_caller.domain.bounds, low_d_func_caller.opt_val,
           low_d_func_caller.raw_opt_pt, **kwargs)

def get_high_d_function_caller_from_description(descr, **kwargs):
  """ Gets a high dimensional function caller from the description. """
  segments = descr.split('-')
  domain_dim = int(segments[1])
  descr_to_func_dict = {'hartmann': lambda: get_hartmann_function_caller(6,),
                        'branin': get_branin_function_caller,
                        'currinexp': get_currin_exp_function_caller,
                        'park1': get_park1_function_caller,
                        'park2': get_park2_function_caller,
                        'borehole': get_borehole_function_caller,
                        'shekel': get_shekel_function_caller}
  low_d_func_caller = descr_to_func_dict[segments[0].lower()](**kwargs)
  return get_high_d_function_caller_from_low_d_func_caller(domain_dim,
                                                           low_d_func_caller, **kwargs)

# Finally, one very convenient wrapper.
def get_syn_function_caller_from_name(function_name, **kwargs):
  """ A very convenient wrapper so that you can just get the function from the name. """
  #pylint: disable=too-many-return-statements
  if function_name.lower() == 'hartmann3':
    return get_hartmann_function_caller(3, descr=function_name, **kwargs)
  elif function_name.lower() == 'hartmann6':
    return get_hartmann_function_caller(6, descr=function_name, **kwargs)
  elif function_name.lower() == 'currinexp':
    return get_currin_exp_function_caller(descr=function_name, **kwargs)
  elif function_name.lower() == 'branin':
    return get_branin_function_caller(descr=function_name, **kwargs)
  elif function_name.lower() == 'borehole':
    return get_borehole_function_caller(descr=function_name, **kwargs)
  elif function_name.lower() == 'shekel':
    return get_shekel_function_caller(descr=function_name, **kwargs)
  elif function_name.lower() == 'park1':
    return get_park1_function_caller(descr=function_name, **kwargs)
  elif function_name.lower() == 'park2':
    return get_park2_function_caller(descr=function_name, **kwargs)
  else:
    return get_high_d_function_caller_from_description(descr=function_name, **kwargs)

