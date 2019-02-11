"""
  Monitors are used to monitor the progress of any algorithm. Here we implement
  the most basic monitor which will be inherited by monitors for specific processes.
  -- kandasamy@cs.cmu.edu
"""

import sys

def get_reporter(reporter):
  """ Returns a reporter based on what was passed as argument. If reporter is already
      a reporter then it is returned. Otherwise,dan appropriate reporter is constructed
      and returned. """
  if isinstance(reporter, str):
    if reporter.lower() == 'default':
      reporter = BasicReporter()
    elif reporter.lower() == 'silent':
      reporter = SilentReporter()
    else:
      raise ValueError('If reporter is string, it should be "default" or "silent".')
  elif hasattr(reporter, 'read'):
    return BasicReporter(reporter)
  elif reporter is None:
    reporter = SilentReporter()
  elif not isinstance(reporter, BasicReporter):
    raise ValueError('Pass either a string, BasicReporter or None for reporter. ')
  return reporter


class BasicReporter(object):
  """ The most basic monitor that implements printing etc. """

  def __init__(self, out=sys.stdout):
    """ Constructor. """
    self.out = out

  def write(self, msg, *_):
    """ Writes a message to stdout. """
    if self.out is not None:
      self.out.write(msg)
      self.out.flush()

  def writeln(self, msg, *args):
    """ Writes a message to stdout with a new line. """
    self.write(msg + '\n', *args)


class SilentReporter(BasicReporter):
  """ This reporter prints nothing. """
  def __init__(self):
    """ Constructor. """
    super(SilentReporter, self).__init__(None)

