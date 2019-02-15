"""
  Some utilities when optimising over neural networks.
  --kandasamy@cs.cmu.edu
"""
# pylint: disable=no-member

# Local
from nas.nasbot.opt.function_caller import FunctionCaller
from nas.nasbot.nn import nn_examples


class NNFunctionCaller(FunctionCaller):
  """ Mostly just a place holder class for a Neural network function caller. """
  pass


# Initial pool of CNNs and MLPs ==========================================================
def get_initial_cnn_pool():
  """ Returns the initial pool for CNNs. """
  vgg_nets = [nn_examples.get_vgg_net(1),
              nn_examples.get_vgg_net(2),
              nn_examples.get_vgg_net(3),
              nn_examples.get_vgg_net(4)]
  blocked_cnns = [nn_examples.get_blocked_cnn(3, 1, 1), # 3
                  nn_examples.get_blocked_cnn(3, 2, 1), # 6
                  nn_examples.get_blocked_cnn(3, 3, 1), # 9
                  nn_examples.get_blocked_cnn(3, 4, 1), # 12
                  nn_examples.get_blocked_cnn(3, 5, 1), # 15
                  nn_examples.get_blocked_cnn(4, 4, 1), # 16
                 ]
  resnet_cnns = [nn_examples.get_resnet_cnn(3, 2, 1)]
  multidepth_cnns = [nn_examples.get_multidepth_cnn_eg2()]
  return vgg_nets + blocked_cnns + resnet_cnns + multidepth_cnns

def get_initial_mlp_pool(class_or_reg):
  """ Returns the initial pool of MLPs. """
  blocked_mlps = [nn_examples.get_blocked_mlp(class_or_reg, 3, 2), # 6
                  nn_examples.get_blocked_mlp(class_or_reg, 4, 2), # 8
                  nn_examples.get_blocked_mlp(class_or_reg, 5, 2), # 10
                  nn_examples.get_blocked_mlp(class_or_reg, 3, 4), # 12
                  nn_examples.get_blocked_mlp(class_or_reg, 6, 2), # 12
                  nn_examples.get_blocked_mlp(class_or_reg, 8, 2), # 16
                  nn_examples.get_blocked_mlp(class_or_reg, 6, 3), # 18
                  nn_examples.get_blocked_mlp(class_or_reg, 10, 2), #20
                  nn_examples.get_blocked_mlp(class_or_reg, 4, 6), #24
                  nn_examples.get_blocked_mlp(class_or_reg, 8, 3), #24
                 ]
#   multidepth_mlps = [nn_examples.get_multidepth_mlp_eg1(class_or_reg),
#                      nn_examples.get_multidepth_mlp_eg2(class_or_reg),
#                     ]
  multidepth_mlps = []
  return blocked_mlps + multidepth_mlps

def get_initial_pool(nn_type):
  """ Returns the initial pool from the neural network type. """
  if nn_type.startswith('cnn'):
    return get_initial_cnn_pool()
  elif nn_type.startswith('mlp'):
    return get_initial_mlp_pool(nn_type[4:])

