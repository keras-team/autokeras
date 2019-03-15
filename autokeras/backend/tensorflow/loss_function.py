import tensorflow as tf


def classification_loss(prediction, label):
    return tf.keras.losses.categorical_crossentropy(label, prediction)


def regression_loss(prediction, target):
    return tf.keras.losses.mean_squared_error(target.float(), prediction)


def binary_classification_loss(prediction, label):
    return tf.keras.losses.binary_crossentropy(label, prediction)
