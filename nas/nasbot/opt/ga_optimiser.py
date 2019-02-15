"""
  Implements Genetic algorithms for black-box optimisation.
  --kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name
# pylint: disable=no-member
# pylint: disable=abstract-method

from numpy.random import choice
# Local imports
from nas.nasbot.opt.blackbox_optimiser import BlackboxOptimiser, blackbox_opt_args
from nas.nasbot.utils.general_utils import sample_according_to_exp_probs
from nas.nasbot.utils.option_handler import get_option_specs, load_options
from nas.nasbot.utils.reporters import get_reporter

ga_specific_opt_args = [
  get_option_specs('num_mutations_per_epoch', False, 50,
    'Number of mutations per epoch.'),
  get_option_specs('num_candidates_to_mutate_from', False, -1,
    'The number of candidates to choose the mutations from.'),
  get_option_specs('fitness_sampler_scaling_const', False, 2,
    'The scaling constant for sampling according to exp_probs.'),
  ]

ga_opt_args = ga_specific_opt_args + blackbox_opt_args


class GAOptimiser(BlackboxOptimiser):
  """ Class for optimisation based on Genetic algorithms. """

  def __init__(self, func_caller, worker_manager, mutation_op, crossover_op=None,
               options=None, reporter=None):
    """ Constructor.
      mutation_op: A function which takes in a list of objects and modifies them.
      crossover_op: A function which takes in two objects and performs a cross-over
                    operation.
      So far we have not implemented cross-over but included here in case we want to
      include it in the future.
      For other arguments, see BlackboxOptimiser
    """
    if options is None:
      reporter = get_reporter(reporter)
      options = load_options(ga_opt_args, reporter=reporter)
    super(GAOptimiser, self).__init__(func_caller, worker_manager, model=None,
                                      options=options, reporter=reporter)
    self.mutation_op = mutation_op
    self.crossover_op = crossover_op
    self.to_eval_points = []

  def _child_set_up(self):
    """ Additional set up. """
    # pylint: disable=attribute-defined-outside-init
    # Set up parameters for the mutations
    self.method_name = 'GA'
    self.num_mutations_per_epoch = self.options.num_mutations_per_epoch
    self.num_candidates_to_mutate_from = self.options.num_candidates_to_mutate_from

  def _child_optimise_initialise(self):
    """ No initialisation for GA. """
    self.generate_new_eval_points()

  def _child_add_data_to_model(self, qinfos):
    """ Update the optimisation model. """
    pass

  def _child_build_new_model(self):
    """ Build new optimisation model. """
    pass

  def _get_candidates_to_mutate_from(self, num_mutations, num_candidates_to_mutate_from):
    """ Returns the candidates to mutate from. """
    all_prev_eval_points = self.pre_eval_points + self.history.query_points
    all_prev_eval_vals = self.pre_eval_vals + self.history.query_vals
    if num_candidates_to_mutate_from <= 0:
      idxs_to_mutate_from = sample_according_to_exp_probs(all_prev_eval_vals,
                              num_mutations, replace=True,
                              scaling_const=self.options.fitness_sampler_scaling_const)
      num_mutations_arg_to_mutation_op = [(idxs_to_mutate_from == i).sum() for i
                                          in range(len(all_prev_eval_points))]
      candidates_to_mutate_from = all_prev_eval_points
    else:
      cand_idxs_to_mutate_from = sample_according_to_exp_probs(all_prev_eval_vals,
        num_candidates_to_mutate_from, replace=False,
        scaling_const=self.options.fitness_sampler_scaling_const)
      candidates_to_mutate_from = [all_prev_eval_points[i] for i in
                                   cand_idxs_to_mutate_from]
      num_mutations_arg_to_mutation_op = num_mutations
    return candidates_to_mutate_from, num_mutations_arg_to_mutation_op

  def generate_new_eval_points(self, num_mutations=None,
                               num_candidates_to_mutate_from=None):
    """ Generates the mutations. """
    num_mutations = self.num_mutations_per_epoch if num_mutations is None else \
                      num_mutations
    num_candidates_to_mutate_from = self.num_candidates_to_mutate_from if \
      num_candidates_to_mutate_from is None else num_candidates_to_mutate_from
    candidates_to_mutate_from, num_mutations_arg_to_mutation_op = \
      self._get_candidates_to_mutate_from(num_mutations, num_candidates_to_mutate_from)
    new_eval_points = self.mutation_op(candidates_to_mutate_from,
                                       num_mutations_arg_to_mutation_op)
    self.to_eval_points.extend(new_eval_points)

  def _determine_next_eval_point(self):
    """ Determine the next point for evaluation. """
    ret = self.to_eval_points.pop(0)
    if len(self.to_eval_points) == 0:
      self.generate_new_eval_points()
    return ret

  def _determine_next_batch_of_eval_points(self):
    """ Determines the next batch of eval points. Not implementing for now. """
    raise NotImplementedError('Not implementing synchronous setting yet.')


# A GA optimiser with random fitness values ----------------------------------------------
class GARandOptimiser(GAOptimiser):
  """ Same as the GA optimiser, but the candidates to mutate from are picked randomly.
      This is used in the RAND baseline.
  """

  def _child_set_up(self):
    """ Additional set up. """
    super(GARandOptimiser, self)._child_set_up()
    self.method_name = 'randGA'

  def _get_candidates_to_mutate_from(self, num_mutations, num_candidates_to_mutate_from):
    """ Returns a random list of points from the evaluations to mutate from. """
    all_prev_eval_points = self.pre_eval_points + self.history.query_points
    candidates_to_mutate_from = choice(all_prev_eval_points,
                                       self.num_candidates_to_mutate_from,
                                       replace=False)
    return candidates_to_mutate_from, num_mutations


# APIs
# ======================================================================================
def ga_optimise_from_args(func_caller, worker_manager, max_capital, mode, mutation_op,
                          is_rand=False, crossover_op=None, options=None,
                          reporter='default'):
  """ GA optimisation from args. """
  if options is None:
    reporter = get_reporter(reporter)
    options = load_options(ga_opt_args, reporter=reporter)
    options.mode = mode
  optimiser_class = GARandOptimiser if is_rand else GAOptimiser
  return (optimiser_class(func_caller, worker_manager, mutation_op, crossover_op,
                          options, reporter)).optimise(max_capital)

