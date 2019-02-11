"""
  A harness to load options.
  -- kandasamy@cs.cmu.edu
"""
# pylint: disable=star-args

import argparse
from copy import deepcopy


def get_option_specs(name, required=False, default=None, help_str='', **kwargs):
  """ A wrapper function to get a specification as a dictionary. """
  ret = {'name':name, 'required':required, 'default':default, 'help':help_str}
  for key, value in kwargs.items():
    ret[key] = value
  return ret

def _print_options(ondp, desc, reporter):
  """ Prints the options out. """
  if reporter is None:
    return
  title_str = 'Hyper-parameters for %s '%(desc)
  title_str = title_str + '-'*(80 - len(title_str))
  reporter.writeln(title_str)
  for key, value in sorted(ondp.items()):
    is_changed_str = '*' if value[0] != value[1] else ' '
    reporter.writeln('  %s %s %s'%(key.ljust(30), is_changed_str, str(value[1])))


def load_options(list_of_options, descr='Algorithm', reporter=None):
  """ Given a list of options, this reads them from the command line and returns
      a namespace with the values.
  """
  parser = argparse.ArgumentParser(description=descr)
  opt_names_default_parsed = {}
  for elem in list_of_options:
    opt_dict = deepcopy(elem)
    opt_name = opt_dict.pop('name')
    opt_names_default_parsed[opt_name] = [opt_dict['default'], None]
    if not opt_name.startswith('--'):
      opt_name = '--' + opt_name
    parser.add_argument(opt_name, **opt_dict)
  args = parser.parse_args()
  for key in opt_names_default_parsed:
    opt_names_default_parsed[key][1] = getattr(args, key)
  _print_options(opt_names_default_parsed, descr, reporter)
  return args

