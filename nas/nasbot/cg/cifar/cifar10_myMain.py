"""
  Defining loss_fn, model_fn, and input_fn for running cnn on cifar10.
  -- willie@cs.cmu.edu
"""

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
# And modified by Willie, 2018.
# ==============================================================================

from __future__ import division
from __future__ import print_function

import argparse
import functools
import itertools
import os

from ..cifar import cifar10
from ..cifar import cifar10_model
from ..cifar import cifar10_utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from ...opt.nn_opt_utils import get_initial_pool

tf.logging.set_verbosity(tf.logging.INFO)


def _loss_fn(is_training, weight_decay, feature, label, data_format,
              nnObj, batch_norm_decay, batch_norm_epsilon):
  """Build loss function for given network.

  Args:
    is_training: true if is training graph.
    weight_decay: weight regularization strength, a float.
    feature: a Tensor.
    label: a Tensor.
    data_format: channels_last (NHWC) or channels_first (NCHW).
    num_layers: number of layers, an int.
    nnObj: neural_network object
    batch_norm_decay: decay for batch normalization, a float.
    batch_norm_epsilon: epsilon for batch normalization, a float.

  Returns:
    A tuple with the loss, the gradients and parameters, and predictions.

  """
  model = cifar10_model.ConvNetCifar10(
      nnObj,
      batch_norm_decay=batch_norm_decay,
      batch_norm_epsilon=batch_norm_epsilon,
      is_training=is_training,
      data_format=data_format)

  logits_sub = model.forward_pass(feature, input_data_format='channels_last')

  # Loss and grad
  tower_loss_sub = [tf.losses.sparse_softmax_cross_entropy(logits=x,
      labels=label) for x in logits_sub] 
  tower_loss = tf.add_n(tower_loss_sub)
  tower_loss = tf.reduce_mean(tower_loss)
  model_params = tf.trainable_variables()
  tower_loss += weight_decay * tf.add_n(
      [tf.nn.l2_loss(v) for v in model_params])
  tower_grad = tf.gradients(tower_loss, model_params)

  # Prediction
  tower_pred = {
      'classes': tf.argmax(input=tf.add_n([tf.nn.softmax(x) for x in logits_sub]),axis=1),
      'probabilities': tf.scalar_mul( tf.Variable(1./len(logits_sub),tf.float32), tf.add_n([tf.nn.softmax(x) for x in logits_sub]) )
  }
  return tower_loss, zip(tower_grad, model_params), tower_pred


def get_model_fn(num_gpus, variable_strategy, num_workers, nnObj):
  """Returns a function that will build the resnet model."""

  def _cnn_model_fn(features, labels, mode, params):
    """Resnet model body.

    Support single host, one or more GPU training. Parameter distribution can
    be either one of the following scheme.
    1. CPU is the parameter server and manages gradient updates.
    2. Parameters are distributed evenly across all GPUs, and the first GPU
       manages gradient updates.

    Args:
      features: a list of tensors, one for each tower
      labels: a list of tensors, one for each tower
      mode: ModeKeys.TRAIN or EVAL
      params: Hyperparameters suitable for tuning
    Returns:
      A EstimatorSpec object.
    """
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    weight_decay = params.weight_decay
    momentum = params.momentum

    tower_features = features
    tower_labels = labels
    tower_losses = []
    tower_gradvars = []
    tower_preds = []

    # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
    # on CPU. The exception is Intel MKL on CPU which is optimal with
    # channels_last.
    data_format = params.data_format
    if not data_format:
      if num_gpus == 0:
        data_format = 'channels_last'
      else:
        data_format = 'channels_first'

    if num_gpus == 0:
      num_devices = 1
      device_type = 'cpu'
    else:
      num_devices = num_gpus
      device_type = 'gpu'

    for i in range(num_devices):
      worker_device = '/{}:{}'.format(device_type, i)
      if variable_strategy == 'CPU':
        device_setter = cifar10_utils.local_device_setter(
            worker_device=worker_device)
      elif variable_strategy == 'GPU':
        device_setter = cifar10_utils.local_device_setter(
            ps_device_type='gpu',
            worker_device=worker_device,
            ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                num_gpus, tf.contrib.training.byte_size_load_fn))
      with tf.variable_scope('cnn', reuse=bool(i != 0)):
        with tf.name_scope('device_%d' % i) as name_scope:
          with tf.device(device_setter):
            loss, gradvars, preds = _loss_fn(
                is_training, weight_decay, tower_features[i], tower_labels[i],
                data_format, nnObj, params.batch_norm_decay,
                params.batch_norm_epsilon)
            tower_losses.append(loss)
            tower_gradvars.append(gradvars)
            tower_preds.append(preds)
            if i == 0:
              # Only trigger batch_norm moving mean and variance update from
              # the 1st tower. Ideally, we should grab the updates from all
              # towers but these stats accumulate extremely fast so we can
              # ignore the other stats from the other towers without
              # significant detriment.
              update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,name_scope)

    # Now compute global loss and gradients.
    gradvars = []
    with tf.name_scope('gradient_averaging'):
      all_grads = {}
      for grad, var in itertools.chain(*tower_gradvars):
        if grad is not None:
          all_grads.setdefault(var, []).append(grad)
      for var, grads in six.iteritems(all_grads):
        # Average gradients on the same device as the variables
        # to which they apply.
        with tf.device(var.device):
          if len(grads) == 1:
            avg_grad = grads[0]
          else:
            avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))
        gradvars.append((avg_grad, var))

    # Device that runs the ops to apply global gradient updates.
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):
      # Suggested learning rate scheduling from
      # https://github.com/ppwwyyxx/tensorpack/blob/master/examples/ResNet/cifar10-resnet.py#L155
      num_batches_per_epoch = cifar10.Cifar10DataSet.num_examples_per_epoch(
          'train') // (params.train_batch_size * num_workers)   # Note: I believe this is 45000/trainBatch, e.g. 45000/20=2250
      ##################################
      # NOTE: The following are old code snippets; either example code originally given, or previous modifications that didn't work as well.
      #boundaries = [
          #num_batches_per_epoch * x
          #for x in np.array([82, 123, 300], dtype=np.int64) # ORIGINAL CODE
          #for x in np.array([27, 100, 200], dtype=np.int64)  # NEW STEP SIZE BOUNDARIES
          #for x in np.array([20, 75, 150], dtype=np.int64)  # NEW STEP SIZE BOUNDARIES , global steps: 45k, 168.75k, 337.5k
          #for x in np.array([30, 50, 100], dtype=np.int64)  # NEW STEP SIZE BOUNDARIES , global steps: 67.5k, 112.5k, 225k
      #]
      #staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]]
      ##################################
      boundaries = [
          num_batches_per_epoch * x
          for x in np.array([15, 40, 80, 120], dtype=np.int64)  # NEW STEP SIZE BOUNDARIES , global steps: 33.75k, 90k, 180k, 270k
      ]
      staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.001, 0.0005]]

      learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
                                                  boundaries, staged_lr)

      loss = tf.reduce_mean(tower_losses, name='loss')

      examples_sec_hook = cifar10_utils.ExamplesPerSecondHook(
          params.train_batch_size, every_n_steps=10)

      tensors_to_log = {'learning_rate': learning_rate, 'loss': loss}

      logging_hook = tf.train.LoggingTensorHook(
          tensors=tensors_to_log, every_n_iter=100)

      train_hooks = [logging_hook, examples_sec_hook]

      optimizer = tf.train.MomentumOptimizer(
          learning_rate=learning_rate, momentum=momentum)

      if params.sync:
        optimizer = tf.train.SyncReplicasOptimizer(
            optimizer, replicas_to_aggregate=num_workers)
        sync_replicas_hook = optimizer.make_session_run_hook(params.is_chief)
        train_hooks.append(sync_replicas_hook)

      # Create single grouped train op
      train_op = [
          optimizer.apply_gradients(
              gradvars, global_step=tf.train.get_global_step())
      ]
      train_op.extend(update_ops)
      train_op = tf.group(*train_op)

      predictions = {
          'classes':
              tf.concat([p['classes'] for p in tower_preds], axis=0),
          'probabilities':
              tf.concat([p['probabilities'] for p in tower_preds], axis=0)
      }
      stacked_labels = tf.concat(labels, axis=0)
      metrics = {
          'accuracy':
              tf.metrics.accuracy(stacked_labels, predictions['classes'])
      }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        training_hooks=train_hooks,
        eval_metric_ops=metrics)

  return _cnn_model_fn


def input_fn(data_dir,
             subset,
             num_shards,
             batch_size,
             use_distortion_for_training=True):
  """Create input graph for model.

  Args:
    data_dir: Directory where TFRecords representing the dataset are located.
    subset: one of 'train', 'validate' and 'eval'.
    num_shards: num of towers participating in data-parallel training.
    batch_size: total batch size for training to be divided by the number of
    shards.
    use_distortion_for_training: True to use distortions.
  Returns:
    two lists of tensors for features and labels, each of num_shards length.
  """
  with tf.device('/cpu:0'):
    use_distortion = subset == 'train' and use_distortion_for_training
    dataset = cifar10.Cifar10DataSet(data_dir, subset, use_distortion)
    image_batch, label_batch = dataset.make_batch(batch_size)
    if num_shards <= 1:
      # No GPU available or only 1 GPU.
      return [image_batch], [label_batch]

    # Note that passing num=batch_size is safe here, even though
    # dataset.batch(batch_size) can, in some cases, return fewer than batch_size
    # examples. This is because it does so only when repeating for a limited
    # number of epochs, but our dataset repeats forever.
    image_batch = tf.unstack(image_batch, num=batch_size, axis=0)
    label_batch = tf.unstack(label_batch, num=batch_size, axis=0)
    feature_shards = [[] for i in range(num_shards)]
    label_shards = [[] for i in range(num_shards)]
    for i in xrange(batch_size):
      idx = i % num_shards
      feature_shards[idx].append(image_batch[i])
      label_shards[idx].append(label_batch[i])
    feature_shards = [tf.parallel_stack(x) for x in feature_shards]
    label_shards = [tf.parallel_stack(x) for x in label_shards]
    return feature_shards, label_shards
