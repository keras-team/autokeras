"""
  Implements a base class for unit tests with some common utilities.
"""

import numpy as np
import random
import sys
from time import time
import unittest


class BaseTestClass(unittest.TestCase):
  """ An abstract base class for unit tests. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(BaseTestClass, self).__init__(*args, **kwargs)

  @classmethod
  def report(cls, msg, msg_type='test_header', out=sys.stdout):
    " Reports a message. """
    prefix = '' # default prefix and suffix
    suffix = '\n'
    if msg_type == 'header':
      suffix = '\n' + '=' * len(msg) + '\n'
    elif msg_type == 'test_header':
      prefix = ' * '
    elif msg_type == 'test_result':
      prefix = '     - '
    out.write(prefix + msg + suffix)
    out.flush()


def execute_tests(seed_val='time'):
  """ Executes the tests. """
  # Set seed value
  # pylint: disable=superfluous-parens
  # pylint: disable=no-member
  if seed_val is not None:
    if seed_val == 'time':
      seed_val = int(time()*10) % 100000
    elif not isinstance(seed_val, int):
      raise ValueError('seed_val should be \'time\', an integer or None.')
    print('Setting random seed to %d.'%(seed_val))
    np.random.seed(seed_val)
    random.seed(seed_val)
  # Run unit tests
  unittest.main()

