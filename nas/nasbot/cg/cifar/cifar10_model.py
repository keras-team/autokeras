"""
  ConvNetCifar10 model class.
  -- willie@cs.cmu.edu
"""

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# And modified by Willie, 2018.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nas.nasbot.cg.cifar.cifar10_model_base import ConvNet


def get_layer_parents(adjList,lidx):
  """ Returns parent layer indices for a given layer index. """
  return [e[0] for e in adjList if e[1]==lidx]


class ConvNetCifar10(ConvNet):
  """Cifar10 dataset with CNN network model."""

  def __init__(self,
               nnObj,
               is_training,
               batch_norm_decay,
               batch_norm_epsilon,
               data_format='channels_first'):
    super(ConvNetCifar10, self).__init__(
        is_training,
        data_format,
        batch_norm_decay,
        batch_norm_epsilon
    )
    self.num_classes = 10 + 1
    self.nnObj = nnObj


  def _get_layers(self,nn,lidx,num_incoming_filters=None):
    # Inputs:
    #   nn - neural_network object
    #   lidx - layer index
    #   num_incoming_filters - number of filters from parents (after concatenation)
    layerStr = nn.layer_labels[lidx]
    strideVal = nn.strides[lidx]
    poolSizeVal = 2
    stridePoolVal = 2
    num_filters = nn.num_units_in_each_layer[lidx]
    if num_filters==None: num_filters=1
    if layerStr=='relu':
      return lambda x: self._relu_layer(x)
    elif layerStr=='conv3':
      return lambda x: self._conv_layer(x,3,num_filters,strideVal)
    elif layerStr=='conv5':
      return lambda x: self._conv_layer(x,5,num_filters,strideVal)
    elif layerStr=='conv7':
      return lambda x: self._conv_layer(x,7,num_filters,strideVal)
    elif layerStr=='conv9':
      return lambda x: self._conv_layer(x,9,num_filters,strideVal)
    elif layerStr=='res3':
      return lambda x: self._residual_layer(x,3,num_incoming_filters,num_filters,strideVal)
    elif layerStr=='res5':
      return lambda x: self._residual_layer(x,5,num_incoming_filters,num_filters,strideVal)
    elif layerStr=='res7':
      return lambda x: self._residual_layer(x,7,num_incoming_filters,num_filters,strideVal)
    elif layerStr=='res9':
      return lambda x: self._residual_layer(x,9,num_incoming_filters,num_filters,strideVal)
    elif layerStr=='avg-pool':
      return lambda x: self._avg_pool_layer(x,poolSizeVal,stridePoolVal)
    elif layerStr=='max-pool':
      return lambda x: self._max_pool_layer(x,poolSizeVal,stridePoolVal)
    elif layerStr=='fc':
      return lambda x: self._fully_connected_layer(x,num_filters)
    elif layerStr=='softmax':
      num_filters=self.num_classes
      return lambda x: self._softmax(x,num_filters)


  def forward_pass(self, x, input_data_format='channels_last'):
    """Build the core model within the graph."""

    if self._data_format != input_data_format:
      if input_data_format == 'channels_last':
        # Computation requires channels_first.
        x = tf.transpose(x, [0, 3, 1, 2])  # Change position of 3 only
      else:
        # Computation requires channels_last.
        x = tf.transpose(x, [0, 2, 3, 1]) # Change position of 1 only

    # Image standardization.
    x = x / 128 - 1
    nn = self.nnObj

    # Printing next architecture before translating into tensorflow
    print('=================================================')
    print('List of layers and num-units in next architecture:')
    for lidx in range(1,nn.num_internal_layers+1):
      layerToPrint = nn.layer_labels[lidx]
      unitsToPrint = nn.num_units_in_each_layer[lidx]
      print('layer-label = %s,  num-units = %s' % (layerToPrint, unitsToPrint))
    print('=================================================')

    # Loop over layers and define conv net 
    layers = [x]  # Add first layer to layers-list 
    for lidx in range(1,nn.num_internal_layers+1):
      # Find and concatenate parent layers
      plist = get_layer_parents(nn.conn_mat.keys(),lidx)
      parent_layers = [layers[i] for i in plist]
      if self._data_format == 'channels_last':
        input_layer = tf.concat(parent_layers,3)
        num_incoming_filters = input_layer.get_shape().as_list()[-1]
      else:
        input_layer = tf.concat(parent_layers,1)
        num_incoming_filters = input_layer.get_shape().as_list()[1]
      # Add layer to layers-list
      nextLayer = self._get_layers(nn,lidx,num_incoming_filters)
      layers.append(nextLayer(input_layer))

    # Define output layer
    plist = get_layer_parents(nn.conn_mat.keys(),lidx+1) # indices for parents of output layer
    parent_layers = [layers[i] for i in plist] # parent layers of output layer
    return parent_layers
