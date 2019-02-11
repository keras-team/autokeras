"""
 Defining, training, and evaluating neural networks in tensorflow.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import os

# To remove the tensorflow compilation warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.logging.set_verbosity(tf.logging.INFO)

activation_dict = {'relu':tf.nn.relu,
                'elu':tf.nn.elu,
                'crelu':tf.nn.crelu,
                'relu6':tf.nn.relu6,
                'softplus':tf.nn.softplus,
                'softmax':tf.nn.softmax, 
                'linear':None,
                'logistic':tf.nn.sigmoid,
                'tanh':tf.nn.tanh,
                'leaky-relu':tf.nn.relu, # Need to update tf for leaky_relu; leaky_relu --> relu.
                'relu-x':tf.nn.relu, # Not sure what relu-x is in tf; relu-x --> relu.
                'step':tf.nn.tanh, # Not sure how to do step in tf; step --> tanh
                }

def get_layer_parents(adjList,lidx):
  """ Returns parent layer indices for a given layer index. """
  # Return all parents (layer indices in list) for layer lidx
  return [e[0] for e in adjList if e[1]==lidx]


def mlp_definition(features,nn,num_classes):
  """ Defines layers in tensorflow neural network, using info from nn python structure. """
  # Define input layer, cast data as tensor
  features = features['x']
  layers = [tf.reshape(tf.cast(features,tf.float32), features.shape)]  ### NEED TO VERIFY FLOAT32 

  # Loop over layers and build tensorflow network
  for lidx in range(1,nn.num_internal_layers+1):
    plist = get_layer_parents(nn.conn_mat.keys(),lidx)
    # Define or concatenate parents
    parent_layers = [layers[i] for i in plist]
    input_layer = tf.concat(parent_layers,1)  ### NEED TO VERIFY CONCAT ALONG AXIS 1
    # Get number of hidden units
    num_units = nn.num_units_in_each_layer[lidx]
    if num_units==None: num_units=1
    # Define activation function
    act_str = nn.layer_labels[lidx]
    # define next layer
    layers.append(tf.layers.dense(input_layer,num_units,use_bias=True,activation=activation_dict[act_str]))

  # Define output layer
  plist = get_layer_parents(nn.conn_mat.keys(),lidx+1)
  parent_layers = [layers[i] for i in plist]
  #scalar_mult = tf.Variable(1./(len(plist)+1),tf.float32) ### NEED TO VERIFY FLOAT 32
  scalar_mult = tf.Variable(1./len(plist),tf.float32) ### NEED TO VERIFY FLOAT 32
  input_layer = tf.scalar_mul(scalar_mult,tf.add_n(parent_layers))
  # For regression
  if nn.class_or_reg=='reg':
    op_layer = tf.layers.dense(input_layer,1,use_bias=True,activation=None)
  # For classification
  elif nn.class_or_reg=='class':
    op_layer = tf.layers.dense(input_layer,num_classes,use_bias=True,activation=tf.nn.softmax)
  else:
    pass
  return op_layer


def get_model_fn(nn,learningRate,num_classes=None):
  """ Sets up loss and optimization details for training and evaluating tensorflow neural network. """

  def my_model_fn(features,labels,mode,nnet,lr,num_classes):
      # Getting tf.estimator.EstimatorSpec for PREDICT, TRAIN, and EVAL modes
      op_layer = mlp_definition(features,nn,num_classes)
      # following predictions are used for multiple modes (PREDICT and EVAL) 
      predictions = {
          "classes": tf.argmax(input=op_layer, axis=1),
          "probabilities": op_layer}
      label_predictions = {
          "classes": tf.argmax(input=labels, axis=1)
      }
      # Make sure labels are correct type
      if nn.class_or_reg=='reg':
        labels = tf.cast(labels,tf.float32)
      elif nn.class_or_reg=='class':
        labels = tf.cast(labels,tf.int32)
      # For mode = PREDICT
      if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
      # Calculate Loss (for both mode = TRAIN and EVAL)
      if nn.class_or_reg=='reg':
        loss = tf.losses.absolute_difference(labels=labels, predictions=op_layer) # Specify loss here 
      elif nn.class_or_reg=='class':
        loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=op_layer) # Specify loss here
        #loss = tf.losses.absolute_difference(labels=labels, predictions=op_layer) # Specify loss here 
        #loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=op_layer) # Specify loss here
      # For mode = TRAIN
      if mode == tf.estimator.ModeKeys.TRAIN:
        # Set learning rate
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learningRate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
      if nn.class_or_reg=='reg':
        eval_metric_ops = {
          "mse": tf.metrics.mean_squared_error(labels=labels, predictions=predictions["probabilities"])
        }
      elif nn.class_or_reg=='class':
        labels = tf.cast(labels,tf.int32)
        eval_metric_ops = {
          "accuracy": tf.metrics.accuracy(labels=label_predictions["classes"], predictions=predictions["classes"])
        }
      return tf.estimator.EstimatorSpec(
          mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

  def _get_my_model_fn(_nn,_learningRate,_num_classes):
      return lambda features, labels, mode: my_model_fn(features,labels,mode,_nn,_learningRate,_num_classes)

  ret = _get_my_model_fn(nn,learningRate,num_classes)
  return ret


def compute_validation_error(mlp,data_train,data_vali,gpu_id,params, tmp_dir):
  """ Trains tensorflow neural network and then computes validation error. """

  # Check if num_classes exists
  if 'num_classes' in params:
      num_classes = params['num_classes']
  else:
      num_classes = None;

  model_fn = get_model_fn(mlp,params['learningRate'],num_classes)
  model = tf.estimator.Estimator(model_fn, model_dir=tmp_dir)
  deviceStr = '/gpu:'+str(gpu_id)
  #deviceStr = '/cpu:'+str(gpu_id)

  # Define input layer (hook in data)
  x_train = np.array(data_train['x'])
  y_train = np.array(data_train['y'])
  if mlp.class_or_reg=='reg':
    y_train = np.reshape(y_train, (y_train.shape[0],1))

  # Validation set
  x_vali = np.array(data_vali['x'])
  y_vali = np.array(data_vali['y'])
  if mlp.class_or_reg=='reg':
    y_vali = np.reshape(y_vali, (y_vali.shape[0],1))

  with tf.device(deviceStr):
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_train},
      y=y_train,
      num_epochs=None,
      shuffle=True,
      batch_size=params['trainBatchSize'])
    vali_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": x_vali},
      y=y_vali,
      num_epochs=None,
      shuffle=False,
      batch_size=params['valiBatchSize'])
    neg_vali_errors = []
    for loop in range(params['numLoops']):
      model.train(train_input_fn,steps=params['trainNumStepsPerLoop'])
      if mlp.class_or_reg=='reg':
        neg_vali_errors.append(-1*model.evaluate(vali_input_fn,steps=params['valiNumStepsPerLoop'])['mse'])
      elif mlp.class_or_reg=='class':
        neg_vali_errors.append(model.evaluate(vali_input_fn,steps=params['valiNumStepsPerLoop'])['accuracy'])
      print('Finished iter: ' + str((loop+1)*params['trainNumStepsPerLoop']))

  return max(neg_vali_errors)
