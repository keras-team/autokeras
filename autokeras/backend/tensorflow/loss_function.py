import tensorflow as tf


def classification_loss(prediction, label):
    return tf.keras.losses.categorical_crossentropy(tf.nn.softmax(prediction), label)  # , from_logits=True


def regression_loss(prediction, target):
    return tf.keras.losses.mean_squared_error(prediction, target.float())


def binary_classification_loss(prediction, label):
    return tf.keras.losses.binary_crossentropy(tf.nn.sigmoid(prediction), label)  # , from_logits=True
