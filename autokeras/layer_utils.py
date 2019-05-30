import tensorflow as tf


def get_global_average_pooling_layer_class(shape):
    return [tf.keras.layers.GlobalAveragePooling1D,
            tf.keras.layers.GlobalAveragePooling2D,
            tf.keras.layers.GlobalAveragePooling3D][len(shape) - 2]


def flatten(output_node):
    if len(output_node.shape) > 5:
        raise ValueError("Expect the input tensor to have less or equal to 5 dimensions, "
                         "but got {shape}".format(shape=output_node.shape))
    # Flatten the input tensor
    # TODO: Add hp.Choice to use Flatten()
    if len(output_node.shape) > 2:
        global_average_pooling = get_global_average_pooling_layer_class(output_node.shape)
        output_node = global_average_pooling()(output_node)
    return output_node