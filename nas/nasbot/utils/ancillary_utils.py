"""
  A collection of utilities for ancillary purposes.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name
# pylint: disable=relative-import

import numpy as np


def compare_dict(dict_1, dict_2):
  """ Compares two dictionaries. """
  # N.B: Taken from stackoverflow:
  # http://stackoverflow.com/questions/4527942/comparing-two-dictionaries-in-python
  dict_1_keys = set(dict_1.keys())
  dict_2_keys = set(dict_2.keys())
  intersect_keys = dict_1_keys.intersection(dict_2_keys)
  added = dict_1_keys - dict_2_keys
  removed = dict_2_keys - dict_1_keys
  modified = {o: (dict_1[o], dict_2[o]) for o in intersect_keys if dict_1[o] != dict_2[o]}
  same = set(o for o in intersect_keys if dict_1[o] == dict_2[o])
  return added, removed, modified, same


# Tests for monotonicity
def is_nondecreasing(arr):
  """ Returns true if the sequence is non-decreasing. """
  return all([x <= y for x, y in zip(arr, arr[1:])])

def is_nonincreasing(arr):
  """ Returns true if the sequence is non-increasing. """
  return all([x >= y for x, y in zip(arr, arr[1:])])

def is_increasing(arr):
  """ Returns true if the sequence is increasing. """
  return all([x < y for x, y in zip(arr, arr[1:])])

def is_decreasing(arr):
  """ Returns true if the sequence is decreasing. """
  return all([x > y for x, y in zip(arr, arr[1:])])

def dicts_are_equal(dict_1, dict_2):
  """ Returns true if dict_1 and dict_2 are equal. """
  added, removed, modified, _ = compare_dict(dict_1, dict_2)
  return len(added) == 0 and len(removed) == 0 and len(modified) == 0


# Print lists as strings
def get_rounded_list(float_list, round_to_decimals=3):
  """ Rounds the list and returns. """
  ret = np.array(float_list).round(round_to_decimals)
  if isinstance(float_list, list):
    ret = list(ret)
  return ret

def get_list_as_str(list_of_objs):
  """ Returns the list as a string. """
  return '[' + ' '.join([str(x) for x in list_of_objs]) + ']'

def get_list_of_floats_as_str(float_list, round_to_decimals=3):
  """ Rounds the list and returns a string representation. """
  float_list = get_rounded_list(float_list, round_to_decimals)
  return get_list_as_str(float_list)


# Some plotting utilities.
def plot_2d_function(func, bounds, x_label='x', y_label='y', title=None):
  """ Plots a 2D function in bounds. """
  # pylint: disable=unused-variable
  dim_grid_size = 20
  x_grid = np.linspace(bounds[0][0], bounds[0][1], dim_grid_size)
  y_grid = np.linspace(bounds[1][0], bounds[1][1], dim_grid_size)
  XX, YY = np.meshgrid(x_grid, y_grid)
  f_vals = func(XX.ravel(), YY.ravel())
  FF = f_vals.reshape(dim_grid_size, dim_grid_size)
  # Create plot
  import matplotlib.pyplot as plt
  from mpl_toolkits.mplot3d import Axes3D
  fig = plt.figure()
  ax = fig.gca(projection='3d')
  ax.plot_surface(XX, YY, FF)
  plt.xlabel(x_label)
  plt.ylabel(y_label)
  if title is not None:
    plt.title(title)
  return fig, ax, plt

