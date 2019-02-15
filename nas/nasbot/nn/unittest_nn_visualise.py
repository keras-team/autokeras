"""
  Unit tests for nn_visualise.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name

import os
# Local imports
from nas.nasbot.nn import nn_visualise
from nas.nasbot.utils.base_test_class import BaseTestClass, execute_tests
from nas.nasbot.nn.unittest_neural_network import generate_cnn_architectures, generate_mlp_architectures

class NNVisualiseTestCase(BaseTestClass):
  """ Contains unit tests for the nn_visualise.py function. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNVisualiseTestCase, self).__init__(*args, **kwargs)
    self.cnns = generate_cnn_architectures()
    self.mlps_reg = generate_mlp_architectures('reg')
    self.mlps_class = generate_mlp_architectures('class')
    self.save_dir = '../scratch/unittest_visualisations/'
    self.save_dir_for_pres = '../scratch/unittest_visualisations_for_pres/'

  def test_cnn_visualisation(self):
    """ Tests visualisation of the NN. """
    self.report('Testing visualisation of cnns.')
    for idx, cnn in enumerate(self.cnns):
      save_file = os.path.join(self.save_dir, 'cnn_%02d'%(idx))
      save_file_for_pres = os.path.join(self.save_dir_for_pres, 'cnn_%02d'%(idx))
      nn_visualise.visualise_nn(cnn, save_file)
      nn_visualise.visualise_nn(cnn, save_file_for_pres, for_pres=True)

  def test_mlp_visualisation(self):
    """ Tests visualisation of the NN. """
    self.report('Testing visualisation of mlps.')
    for idx in range(len(self.mlps_reg)):
      # For regression
      reg_save_file = os.path.join(self.save_dir, 'mlp_reg_%02d'%(idx))
      reg_save_file_for_pres = os.path.join(self.save_dir_for_pres, 'mlp_reg_%02d'%(idx))
      nn_visualise.visualise_nn(self.mlps_reg[idx], reg_save_file)
      nn_visualise.visualise_nn(self.mlps_reg[idx], reg_save_file_for_pres, for_pres=True)
      # For classification
      cla_save_file = os.path.join(self.save_dir, 'mlp_cla_%02d'%(idx))
      cla_save_file_for_pres = os.path.join(self.save_dir_for_pres, 'mlp_cla_%02d'%(idx))
      nn_visualise.visualise_nn(self.mlps_class[idx], cla_save_file)
      nn_visualise.visualise_nn(self.mlps_class[idx], cla_save_file_for_pres,
                                for_pres=True)


if __name__ == '__main__':
  execute_tests()

