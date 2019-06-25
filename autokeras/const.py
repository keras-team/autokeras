import tensorflow as tf


class Constant(object):
    NUM_TRAILS = 100
    LOWER_BETTER = [tf.keras.metrics.mean_squared_error.__name__,
                    tf.keras.metrics.mean_absolute_error.__name__]
    VALIDATION_SET_SIZE = 0.08333
    # TODO: Change it to random and configurable.
    SEED = 42
    BATCH_SIZE = 128
    RNN_LAYERS = {
        'vanilla': tf.keras.layers.SimpleRNN,
        'gru': tf.keras.layers.GRU,
        'lstm': tf.keras.layers.LSTM
    }
