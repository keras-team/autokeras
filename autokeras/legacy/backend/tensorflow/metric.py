import tensorflow as tf


def classification_metric(prediction, target):
    return tf.keras.metrics.categorical_accuracy


def regression_metric(prediction, target):
    return tf.keras.metrics.mean_squared_error


def binary_classification_metric(prediction, target):
    return tf.keras.metrics.binary_accuracy
