"""
  ConvNet model class.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ConvNet(object):
  """ConvNet model."""

  def __init__(self, is_training, data_format, batch_norm_decay, batch_norm_epsilon):
    """ConvNet constructor.

    Args:
      is_training: if build training or inference model.
      data_format: the data_format used during computation.
                   one of 'channels_first' or 'channels_last'.
    """
    self._batch_norm_decay = batch_norm_decay
    self._batch_norm_epsilon = batch_norm_epsilon
    self._is_training = is_training
    assert data_format in ('channels_first', 'channels_last')
    self._data_format = data_format

  def forward_pass(self, x):
    raise NotImplementedError(
        'forward_pass() is implemented in ConvNet sub classes')

  def _relu(self, x):
    return tf.nn.relu(x)

  def _relu_layer(self, x):
    with tf.name_scope('relu') as name_scope:
      x = self._relu(x)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _conv(self, x, kernel_size, filters, strides, is_atrous=False):
    """Convolution."""

    padding = 'SAME'
    if not is_atrous and strides > 1:
      pad = kernel_size - 1
      pad_beg = pad // 2
      pad_end = pad - pad_beg
      if self._data_format == 'channels_first':
        x = tf.pad(x, [[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
      else:
        x = tf.pad(x, [[0, 0], [pad_beg, pad_end], [pad_beg, pad_end], [0, 0]])
      padding = 'VALID'
    return tf.layers.conv2d(
        inputs=x,
        kernel_size=kernel_size,
        filters=filters,
        strides=strides,
        padding=padding,
        use_bias=False,
        data_format=self._data_format)

  def _myConv(self, x, kernel_size, filters, strides, is_atrous=False):
    """My convolution: conv, batch-norm, relu."""
    x = self._conv(x,kernel_size,filters,strides)
    x = self._batch_norm(x)
    x = self._relu(x)
    return x

  def _conv_layer(self, x, kernel_size, filters, strides, is_atrous=False):
    """Wrapper on _myConv layer, when convolution is used by-itself as a layer."""
    with tf.name_scope('conv_layer') as name_scope:
      x = self._myConv(x, kernel_size, filters, strides, is_atrous=False)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _batch_norm(self, x):
    if self._data_format == 'channels_first':
      data_format = 'NCHW'
    else:
      data_format = 'NHWC'
    return tf.contrib.layers.batch_norm(
        x,
        decay=self._batch_norm_decay,
        center=True,
        scale=True,
        epsilon=self._batch_norm_epsilon,
        is_training=self._is_training,
        fused=True,
        data_format=data_format)

  def _myFullyConnected(self, x, out_dim):
    x = tf.layers.dense(x, out_dim)
    x = self._relu(x)
    return x

  def _fully_connected_layer(self, x, out_dim):
    """Wrapper on _myFullyConnected, when fully connected is used by-itself as a layer."""
    with tf.name_scope('fully_connected') as name_scope:
      if x.get_shape().ndims == 4:
        x = self._myGlobalAvgPool(x) # Flatten
      x = self._myFullyConnected(x, out_dim)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _softmax(self, x, num_units):
    with tf.name_scope('softmax') as name_scope:
      if x.get_shape().ndims == 4:
        x = self._myGlobalAvgPool(x) # Flatten
      x = self._myFullyConnected(x,num_units) # With the tf.argmax/tf.softmax in cifar10_main._tower_fn, this should be softmax layer
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _myAvgPool(self, x, pool_size, stride):
    x = tf.layers.average_pooling2d(
        x, pool_size, stride, 'SAME', data_format=self._data_format)
    return x

  def _avg_pool_layer(self, x, pool_size, stride):
    """Wrapper on _myAvgPool, when average pool is used by-itself as a layer."""
    with tf.name_scope('avg_pool') as name_scope:
      x = self._myAvgPool(x, pool_size, stride)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _myGlobalAvgPool(self, x):
    assert x.get_shape().ndims == 4
    if self._data_format == 'channels_first':
      x = tf.reduce_mean(x, [2, 3])
    else:
      x = tf.reduce_mean(x, [1, 2])
    return x

  def _global_avg_pool_layer(self, x):
    with tf.name_scope('global_avg_pool') as name_scope:
      x = self._myGlobalAvgPool(x)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _max_pool_layer(self, x, pool_size, stride):
    with tf.name_scope('max_pool') as name_scope:
      x = tf.layers.max_pooling2d(
          x, pool_size, stride, 'SAME', data_format=self._data_format)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x

  def _residual_layer(self,
                   x,
                   kernel_size,
                   in_filter,
                   out_filter,
                   stride,
                   activate_before_residual=False):
    """Residual unit with 2 sub layers, using Plan A for shortcut connection."""

    del activate_before_residual
    with tf.name_scope('residual_layer') as name_scope:
      in_filter = int(in_filter)
      out_filter = int(out_filter)
      orig_x = x
      x = self._myConv(x,kernel_size,out_filter,stride)
      x = self._conv(x, kernel_size, out_filter, 1)
      x = self._batch_norm(x)

      # Pad for different number of filters
      if out_filter != in_filter:
        pad_diff = abs(out_filter - in_filter)
        pad1 = pad_diff // 2
        pad2 = pad_diff - pad1
        if out_filter > in_filter:
          orig_x = self._myAvgPool(orig_x, stride, stride)
          if self._data_format == 'channels_first':
            orig_x = tf.pad(orig_x, [[0, 0], [pad1, pad2], [0, 0], [0, 0]])
          else:
            orig_x = tf.pad(orig_x, [[0, 0], [0, 0], [0, 0], [pad1, pad2]])
        elif in_filter > out_filter:
          x = self._myAvgPool(x, stride, stride)
          if self._data_format == 'channels_first':
            x = tf.pad(x, [[0, 0], [pad1, pad2], [0, 0], [0, 0]])
          else:
            x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [pad1, pad2]])

      # Also pad for different image heights and widths (due to pooling)
      orig_xshape = orig_x.get_shape().as_list()
      xshape = x.get_shape().as_list()
      if self._data_format == 'channels_first':
        # Pad height, case: channels_first
        oldHeight = int(orig_xshape[2])
        newHeight = int(xshape[2])
        pad_height_diff = abs(newHeight - oldHeight)
        pad_height1 = pad_height_diff // 2
        pad_height2 = pad_height_diff - pad_height1
        if newHeight < oldHeight:
          x = tf.pad(x,[[0,0], [0,0], [pad_height1,pad_height2], [0,0]])
        elif oldHeight < newHeight:
          orig_x = tf.pad(orig_x,[[0,0], [0,0], [pad_height1,pad_height2], [0,0]])
        # Pad width, case: channels_first
        oldWidth = int(orig_xshape[3])
        newWidth = int(xshape[3])
        pad_width_diff = abs(newWidth - oldWidth)
        pad_width1 = pad_width_diff // 2
        pad_width2 = pad_width_diff - pad_width1
        if newWidth < oldWidth:
          x = tf.pad(x,[[0,0], [0,0], [0,0], [pad_width1,pad_width2]])
        elif oldWidth < newWidth:
          orig_x = tf.pad(orig_x,[[0,0], [0,0], [0,0], [pad_width1,pad_width2]])
      else:
        # Pad height, case: channels_last
        oldHeight = int(orig_xshape[1])
        newHeight = int(xshape[1])
        pad_height_diff = abs(newHeight - oldHeight)
        pad_height1 = pad_height_diff // 2
        pad_height2 = pad_height_diff - pad_height1
        if newHeight < oldHeight:
          x = tf.pad(x,[[0,0], [pad_height1,pad_height2], [0,0], [0,0]])
        elif oldHeight < newHeight:
          orig_x = tf.pad(orig_x,[[0,0], [pad_height1,pad_height2], [0,0], [0,0]])
        # Pad width, case: channels_last
        oldWidth = int(orig_xshape[2])
        newWidth = int(xshape[2])
        pad_width_diff = abs(newWidth - oldWidth)
        pad_width1 = pad_width_diff // 2
        pad_width2 = pad_width_diff - pad_width1
        if newWidth < oldWidth:
          x = tf.pad(x,[[0,0], [0,0], [pad_width1,pad_width2], [0,0]])
        elif oldWidth < newWidth:
          orig_x = tf.pad(orig_x,[[0,0], [0,0], [pad_width1,pad_width2], [0,0]])

      x = self._relu(tf.add(x, orig_x))
      tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
      return x

  def _concat(self, x1, x2):
    with tf.name_scope('concat') as name_scope:
      x = tf.concat([x1,x2],1)
    tf.logging.info('image after unit %s: %s', name_scope, x.get_shape())
    return x
