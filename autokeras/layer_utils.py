import tensorflow as tf
from sklearn import model_selection
from tensorflow.python.util import nest

from autokeras import const


def get_global_average_pooling_layer_class(shape):
    return [tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalAveragePooling3D][len(shape) - 3]


def format_inputs(inputs, name=None, num=None):
    inputs = nest.flatten(inputs)
    if not isinstance(inputs, list):
        inputs = [inputs]

    if num is None:
        return inputs

    if not len(inputs) == num:
        raise ValueError('Expected {num} elements in the '
                         'inputs list for {name} '
                         'but received {len} inputs.'.format(num=num, name=name, len=len(inputs)))
    return inputs


def get_rnn_block(choice):
    return const.Constant.RNN_LAYERS[choice]


def attention_block(inputs):
    time_steps = int(inputs.shape[1])
    attention_out = tf.keras.layers.Permute((2,1))(inputs)
    attention_out = tf.keras.layers.Dense(time_steps, activation='softmax')(attention_out)
    attention_out = tf.keras.layers.Permute((2,1))(attention_out)
    mul_attention_out = tf.keras.layers.Multiply()([inputs, attention_out])
    return mul_attention_out


def seq2seq_builder(inputs,rnn_type,choice_of_layers,feature_size):
    print("In seq2seq ..",inputs.shape)

    block = get_rnn_block(rnn_type)
    # TODO: Autoencoder setup exists. Must accommodate NMT setup in future
    encoder_inputs = decoder_inputs = inputs

    # TODO: Accept different num_layers for encoder and decoder
    for i in range(choice_of_layers):
        return_sequences = False if i == choice_of_layers - 1 else True
        lstm_enc = block(feature_size, return_state=True,return_sequences=return_sequences)
        encoder_inputs = lstm_enc(encoder_inputs)
        print(len(encoder_inputs),encoder_inputs[0].shape,encoder_inputs[1].shape)
    if rnn_type == 'lstm':
        enc_out, state_h, state_c = encoder_inputs
        encoder_states = [state_h, state_c]
    else:
        enc_out, state_h = encoder_inputs
        encoder_states = [state_h]

    for i in range(choice_of_layers):
        initial_state = encoder_states if i == 0 else None
        decoder_inputs = block(feature_size, return_state=True, return_sequences=True)(decoder_inputs, initial_state=initial_state)

    dec_out = decoder_inputs[0]

    return dec_out


def split_train_to_valid(x, y):
    # Generate split index
    validation_set_size = int(len(x[0]) * const.Constant.VALIDATION_SET_SIZE)
    validation_set_size = min(validation_set_size, 500)
    validation_set_size = max(validation_set_size, 1)
    train_index, valid_index = model_selection.train_test_split(range(len(x[0])),
                                                                test_size=validation_set_size,
                                                                random_state=const.Constant.SEED)

    # Split the data
    x_train = None
    y_train = None
    x_val = None
    y_val = None
    for temp_x_train_input in x:
        x_train, x_val = temp_x_train_input[train_index], temp_x_train_input[valid_index]
    for temp_y_train_input in y:
        y_train, y_val = temp_y_train_input[train_index], temp_y_train_input[valid_index]

    return (x_train, y_train), (x_val, y_val)


def get_name_scope():
    with tf.name_scope('a') as scope:
        name_scope = scope[:-2]
    return name_scope
