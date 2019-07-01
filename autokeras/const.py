import tensorflow as tf


class Constant(object):
    TEMP_DIRECTORY = './tmp'
    NUM_TRAILS = 100
    VALIDATION_SET_SIZE = 0.08333
    # TODO: Change it to random and configurable.
    SEED = 42
    BATCH_SIZE = 128
    RNN_LAYERS = {
        'vanilla': tf.keras.layers.SimpleRNN,
        'gru': tf.keras.layers.GRU,
        'lstm': tf.keras.layers.LSTM
    }
    S2S_TYPES = [
        'auto_enc',
        'nmt',
        'text_gen',
    ]
