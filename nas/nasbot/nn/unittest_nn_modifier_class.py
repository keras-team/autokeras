"""
  Unit tests for the class NNModifier in nn_modifiers.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=invalid-name

from copy import deepcopy
import numpy as np
import os
from shutil import rmtree
# Local imports
from ..nn import nn_constraint_checkers
from ..nn import nn_modifiers
from ..nn.neural_network import NeuralNetwork
from ..nn.nn_visualise import visualise_nn
from .unittest_neural_network import generate_cnn_architectures, generate_mlp_architectures
from ..utils.base_test_class import BaseTestClass, execute_tests


def test_if_two_networks_are_equal(net1, net2, false_if_net1_is_net2=True):
  """ Returns true if both net1 and net2 are equal.
      If any part of net1 is copied onto net2, then the output will be false
      if false_if_net1_is_net2 is True (default).
  """
  is_true = True
  for key in net1.__dict__.keys():
    val1 = net1.__dict__[key]
    val2 = net2.__dict__[key]
    is_true = True
    if isinstance(val1, dict):
      if false_if_net1_is_net2:
        is_true = is_true and (val1 is not val2)
      for val_key in val1.keys():
        is_true = is_true and np.all(val1[val_key] == val2[val_key])
    elif hasattr(val1, '__iter__'):
      if false_if_net1_is_net2:
        is_true = is_true and (val1 is not val2)
      is_true = is_true and np.all(val1 == val2)
    else:
      is_true = is_true and val1 == val2
    if not is_true: # break here if necessary
      return is_true
  return is_true


def test_for_orig_vs_modifications(save_dir, save_prefix, old_nn,
                                   get_modifications, constraint_checker, write_result):
  """ Tests for the original network and the modifications. Also, visualises the networks.
  """
  visualise_nn(old_nn, os.path.join(save_dir, '%s_orig'%(save_prefix)))
  old_nn_copy = deepcopy(old_nn)
  # Get the modified networks.
  new_nns = get_modifications(old_nn)
  # Go through each new network.
  for new_idx, new_nn in enumerate(new_nns):
    assert isinstance(new_nn, NeuralNetwork)
    assert constraint_checker(new_nn)
    visualise_nn(new_nn, os.path.join(save_dir, '%s_%d'%(save_prefix, new_idx)))
  # Finally test if the networks have not changed.
  assert test_if_two_networks_are_equal(old_nn, old_nn_copy)
  write_result('%s (%s):: #new-networks: %d.'%(
    save_prefix, old_nn.nn_class, len(new_nns)), 'test_result')


class NNModifierTestCase(BaseTestClass):
  """ Unit tests for the NNModifier class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNModifierTestCase, self).__init__(*args, **kwargs)
    self.cnns = generate_cnn_architectures()
    self.mlps = generate_mlp_architectures()
    self.save_dir = '../scratch/unittest_modifier_class/'
    self.cnn_constraint_checker = nn_constraint_checkers.CNNConstraintChecker(
      50, 4, np.inf, 4.0, 5, 5, 100, 8000, 8)
    self.mlp_constraint_checker = nn_constraint_checkers.MLPConstraintChecker(
      50, 4, np.inf, 4.0, 5, 5, 100, 8000, 8)
    self.cnn_modifier = nn_modifiers.NNModifier(self.cnn_constraint_checker)
    self.mlp_modifier = nn_modifiers.NNModifier(self.mlp_constraint_checker)
    self.modifier_wo_cc = nn_modifiers.NNModifier(None)

  def _get_modifier_and_cc(self, nn):
    """ Returns modifier for the neural network nn."""
    if nn.nn_class == 'cnn':
      modifier = self.cnn_modifier
      constraint_checker = self.cnn_constraint_checker
    else:
      modifier = self.mlp_modifier
      constraint_checker = self.mlp_constraint_checker
    return modifier, constraint_checker

  def test_get_primitives(self):
    """ Test for the get_primitives_grouped_by_type method. """
    self.report('Testing get_primitives_grouped_by_type')
    test_nns = self.cnns + self.mlps
    primitives, _ = self.cnn_modifier.get_primitives_grouped_by_type(self.cnns[0])
    self.report('Types of primitives: %s'%(primitives.keys()), 'test_result')
    for idx, nn in enumerate(test_nns):
      nn_copy = deepcopy(nn)
      modifier, _ = self._get_modifier_and_cc(nn)
      primitives, _ = modifier.get_primitives_grouped_by_type(nn)
      report_str = '%d (%s n=%d,m=%d):: '%(idx, nn.nn_class, nn.num_layers,
                                           nn.get_total_num_edges())
      total_num_primitives = 0
      for _, list_or_prims in primitives.iteritems():
        report_str += '%d, '%(len(list_or_prims))
        total_num_primitives += len(list_or_prims)
      report_str += 'tot=%d'%(total_num_primitives)
      self.report(report_str, 'test_result')
      assert test_if_two_networks_are_equal(nn_copy, nn)

  def test_get_single_step_modifications(self):
    """ Tests single step modifications. """
    self.report('Testing single step modifications.')
    save_dir = os.path.join(self.save_dir, 'single_step')
    if os.path.exists(save_dir):
      rmtree(save_dir)
    # Now iterate through the test networks
    test_nns = self.cnns + self.mlps
    for idx, old_nn in enumerate(test_nns):
      save_prefix = str(idx)
      modifier, constraint_checker = self._get_modifier_and_cc(old_nn)
      if idx in [2, 12]:
        num_modifications = 'all'
      else:
        num_modifications = 'all'
      get_modifications = lambda arg_nn: modifier.get_single_step_modifications(
                                           arg_nn, num_modifications)
      test_for_orig_vs_modifications(save_dir, save_prefix, old_nn,
        get_modifications, constraint_checker, self.report)

  def test_multi_step_modifications(self):
    """ Tests multi step modifications. """
    num_steps = 4
    self.report('Testing %d-step modifications.'%(num_steps))
    num_modifications = 20
    save_dir = os.path.join(self.save_dir, 'multi_step_%d'%(num_steps))
    if os.path.exists(save_dir):
      rmtree(save_dir)
    # Now iterate through the test networks
    test_nns = self.cnns + self.mlps
    for idx, old_nn in enumerate(test_nns):
      save_prefix = str(idx)
      modifier, constraint_checker = self._get_modifier_and_cc(old_nn)
      get_modifications = lambda arg_nn: modifier.get_multi_step_modifications(
                                           arg_nn, num_steps, num_modifications)
      test_for_orig_vs_modifications(save_dir, save_prefix, old_nn,
        get_modifications, constraint_checker, self.report)

  def test_call(self):
    """ Tests the __call__ function with a single input of the modifier. """
    self.report('Testing the __call__ function with single input of the modifier.')
    num_modifications = 20
    num_steps_probs = [0.5, 0.25, 0.125, 0.075, 0.05]
    save_dir = os.path.join(self.save_dir, 'modifier_call_single')
    if os.path.exists(save_dir):
      rmtree(save_dir)
    test_nns = self.cnns + self.mlps
    for idx, old_nn in enumerate(test_nns):
      save_prefix = str(idx)
      modifier, constraint_checker = self._get_modifier_and_cc(old_nn)
      get_modifications = lambda arg_nn: modifier(arg_nn, num_modifications,
                                                  num_steps_probs)
      test_for_orig_vs_modifications(save_dir, save_prefix, old_nn,
        get_modifications, constraint_checker, self.report)

  def test_call_with_list(self):
    """ Tests the __call__ function with a single input of the modifier. """
    self.report('Testing the __call__ function with a list of inputs.')
    num_modifications = 40
    num_steps_probs = [0.5, 0.25, 0.125, 0.075, 0.05]
    save_dir = os.path.join(self.save_dir, 'modifier_call_list')
    if os.path.exists(save_dir):
      rmtree(save_dir)
    test_probs = [self.cnns, self.mlps, generate_mlp_architectures('class')]
    for idx, prob in enumerate(test_probs):
      save_prefix = str(idx)
      modifier = self.modifier_wo_cc
      modifications = modifier(prob, num_modifications, num_steps_probs)
      for new_idx, new_nn in enumerate(modifications):
        assert isinstance(new_nn, NeuralNetwork)
        visualise_nn(new_nn, os.path.join(save_dir, '%s_%d'%(save_prefix, new_idx)))
      self.report('With list of %d nns(%s):: #new-networks: %d.'%(
                   len(prob), prob[0].nn_class, len(modifications)), 'test_result')

if __name__ == '__main__':
  execute_tests()

