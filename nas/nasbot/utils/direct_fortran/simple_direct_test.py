
import numpy as np
import direct

def main():
  """ Main function. """
  obj = lambda x: (np.dot(x-0.1,x), 0)
  lower_bounds = [-1] * 4
  upper_bounds = [1] * 4;
  dim = len(lower_bounds)
  eps = 1e-5
  max_func_evals = 1000
  max_iterations = max_func_evals
  algmethod = 0
#   _log_file = 'dir_file_name'
  _log_file = ''
  fglobal = -1e100
  fglper = 0.01
  volper = -1.0
  sigmaper = -1.0
#   user_data = None

  def _objective_wrap(x, iidata, ddata, cdata, n, iisize, idsize, icsize):
    """
      A wrapper to comply with the fortran requirements.
    """
    return obj(x)

  iidata = np.ones(0, dtype=np.int32)
  ddata = np.ones(0, dtype=np.float64)
  cdata = np.ones([0, 40], dtype=np.uint8)
  

  soln = direct.direct(_objective_wrap,
                       eps,
                       max_func_evals,
                       max_iterations,
                       np.array(lower_bounds, dtype=np.float64),
                       np.array(upper_bounds, dtype=np.float64),
                       algmethod,
                       _log_file,
                       fglobal,
                       fglper,
                       volper,
                       sigmaper,
                       iidata,
                       ddata,
                       cdata
                      )
  print(soln)
                    



if __name__ == '__main__':
  main()

