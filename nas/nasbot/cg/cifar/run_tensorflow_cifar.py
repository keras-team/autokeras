"""
  Defining, training, and evaluating neural networks in tensorflow on CIFAR-10.
  -- willie@cs.cmu.edu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools
import os
import argparse
import tensorflow as tf
from ..cifar import cifar10_myMain

from ...opt.nn_opt_utils import get_initial_pool


os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # To remove the tensorflow compilation warnings
os.environ['TF_SYNC_ON_FINISH'] = '0'
os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'


def compute_validation_error(nn,data_dir,gpu_id,params,tmp_dir):
  """ Trains tensorflow neural network and then computes validation error. """

  # Fixed variables for CIFAR10 CNN runs
  
  if gpu_id < 0:
    variable_strategy = 'CPU'; num_gpus = 0 # For CPU
  else:
    variable_strategy = 'GPU'; num_gpus = 1 # For GPU

  use_distortion_for_training = True
  log_device_placement = False
  num_intra_threads = 0
  weight_decay = 2e-4
  momentum = 0.9
  learning_rate = params['learningRate']
  data_format = None
  batch_norm_decay = 0.997
  batch_norm_epsilon = 1e-5

  hparams = argparse.Namespace(weight_decay=weight_decay, momentum=momentum,
      learning_rate=learning_rate, data_format=data_format,
      batch_norm_decay=batch_norm_decay, batch_norm_epsilon=batch_norm_epsilon,
      train_batch_size=params['trainBatchSize'],
      eval_batch_size=params['valiBatchSize'], sync=False)

  # Get model_fn
  model_fn = cifar10_myMain.get_model_fn(num_gpus, variable_strategy, 1, nn)
  
  # Set model = tf.estimator.Estimator(model_fn)
  model = tf.estimator.Estimator(model_fn, model_dir=tmp_dir, params=hparams)

  # Define train_input_fn and vali_input_fn
  train_input_fn = functools.partial(
      cifar10_myMain.input_fn,
      data_dir,
      subset='train',
      num_shards=num_gpus,
      batch_size=params['trainBatchSize'],
      use_distortion_for_training=use_distortion_for_training)

  vali_input_fn = functools.partial(
      cifar10_myMain.input_fn,
      data_dir,
      subset='validation',
      batch_size=params['valiBatchSize'],
      num_shards=num_gpus)

  # Loop through numLoops: call model.train, and model.evaluate
  neg_vali_errors = []
  for loop in range(params['numLoops']):
    model.train(train_input_fn,steps=params['trainNumStepsPerLoop'])
    neg_vali_errors.append(model.evaluate(vali_input_fn,steps=params['valiNumStepsPerLoop'])['accuracy'])
    print('Finished iter: ' + str((loop+1)*params['trainNumStepsPerLoop']))

  # Print all validation errors and test errors
  print('List of validation accuracies:')
  print(neg_vali_errors)

  return max(neg_vali_errors)
