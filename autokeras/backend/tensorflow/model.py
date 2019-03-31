from tensorflow.keras import layers, models
from copy import deepcopy

from autokeras.nn.layers import StubAdd, StubConcatenate, StubConv1d, StubConv2d, StubConv3d, \
    StubDropout1d, StubDropout2d, StubDropout3d, StubGlobalPooling1d, StubGlobalPooling2d, StubGlobalPooling3d, \
    StubPooling1d, StubPooling2d, StubPooling3d, StubBatchNormalization1d, StubBatchNormalization2d, \
    StubBatchNormalization3d, StubSoftmax, StubReLU, StubFlatten, StubDense, StubAvgPooling1d, StubAvgPooling2d, \
    StubAvgPooling3d


def produce_model(graph):
    KerasModel(graph).model.summary()
    return KerasModel(graph)


class KerasModel:
    """A neural network class using tensorflow keras constructed from an instance of Graph."""

    def __init__(self, graph):
        self.graph = graph
        self.layers = []
        for layer in graph.layer_list:
            self.layers.append(to_real_keras_layer(layer))

        # Construct the keras graph.
        # Input
        topo_node_list = self.graph.topological_order
        output_id = topo_node_list[-1]
        input_id = topo_node_list[0]
        input_tensor = layers.Input(shape=graph.node_list[input_id].shape)

        node_list = deepcopy(self.graph.node_list)
        node_list[input_id] = input_tensor

        # Output
        for v in topo_node_list:
            for u, layer_id in self.graph.reverse_adj_list[v]:
                layer = self.graph.layer_list[layer_id]
                keras_layer = self.layers[layer_id]

                if isinstance(layer, (StubAdd, StubConcatenate)):
                    edge_input_tensor = list(map(lambda x: node_list[x],
                                                 self.graph.layer_id_to_input_node_ids[layer_id]))
                else:
                    edge_input_tensor = node_list[u]

                temp_tensor = keras_layer(edge_input_tensor)
                node_list[v] = temp_tensor

        output_tensor = node_list[output_id]
        self.model = models.Model(inputs=input_tensor, outputs=output_tensor)

        if graph.weighted:
            for index, layer in enumerate(self.layers):
                set_stub_weight_to_keras(self.graph.layer_list[index], layer)

    def set_weight_to_graph(self):
        self.graph.weighted = True
        for index, layer in enumerate(self.layers):
            set_keras_weight_to_stub(layer, self.graph.layer_list[index])


def set_keras_weight_to_stub(keras_layer, stub_layer):
    stub_layer.import_weights_keras(keras_layer)


def set_stub_weight_to_keras(stub_layer, keras_layer):
    stub_layer.export_weights_keras(keras_layer)


def to_real_keras_layer(stub_layer):
    if isinstance(stub_layer, StubConv1d):
        return layers.Conv1D(stub_layer.filters,
                             stub_layer.kernel_size,
                             strides=stub_layer.stride,
                             input_shape=stub_layer.input.shape,
                             padding='same')  # padding

    elif isinstance(stub_layer, StubConv2d):
        return layers.Conv2D(stub_layer.filters,
                             stub_layer.kernel_size,
                             strides=stub_layer.stride,
                             input_shape=stub_layer.input.shape,
                             padding='same')  # padding

    elif isinstance(stub_layer, StubConv3d):
        return layers.Conv3D(stub_layer.filters,
                             stub_layer.kernel_size,
                             strides=stub_layer.stride,
                             input_shape=stub_layer.input.shape,
                             padding='same')  # padding

    # TODO: Spatial Dropout
    elif isinstance(stub_layer, (StubDropout1d, StubDropout2d, StubDropout3d)):
        return layers.Dropout(stub_layer.rate)
    # elif isinstance(stub_layer, StubDropout2d):
    #     return layers.SpatialDropout2D(stub_layer.rate)
    # elif isinstance(stub_layer, StubDropout3d):
    #     return layers.SpatialDropout3D(stub_layer.rate)

    elif isinstance(stub_layer, StubAvgPooling1d):
        return layers.AveragePooling1D(stub_layer.kernel_size, strides=stub_layer.stride)
    elif isinstance(stub_layer, StubAvgPooling2d):
        return layers.AveragePooling2D(stub_layer.kernel_size, strides=stub_layer.stride)
    elif isinstance(stub_layer, StubAvgPooling3d):
        return layers.AveragePooling3D(stub_layer.kernel_size, strides=stub_layer.stride)

    elif isinstance(stub_layer, StubGlobalPooling1d):
        return layers.GlobalAveragePooling1D()
    elif isinstance(stub_layer, StubGlobalPooling2d):
        return layers.GlobalAveragePooling2D()
    elif isinstance(stub_layer, StubGlobalPooling3d):
        return layers.GlobalAveragePooling3D()

    elif isinstance(stub_layer, StubPooling1d):
        return layers.MaxPooling1D(stub_layer.kernel_size, strides=stub_layer.stride)
    elif isinstance(stub_layer, StubPooling2d):
        return layers.MaxPooling2D(stub_layer.kernel_size, strides=stub_layer.stride)
    elif isinstance(stub_layer, StubPooling3d):
        return layers.MaxPooling3D(stub_layer.kernel_size, strides=stub_layer.stride)

    elif isinstance(stub_layer, (StubBatchNormalization1d, StubBatchNormalization2d, StubBatchNormalization3d)):
        return layers.BatchNormalization(input_shape=stub_layer.input.shape)

    elif isinstance(stub_layer, StubSoftmax):
        return layers.Activation('softmax')
    elif isinstance(stub_layer, StubReLU):
        return layers.Activation('relu')
    elif isinstance(stub_layer, StubFlatten):
        return layers.Flatten()
    elif isinstance(stub_layer, StubAdd):
        return layers.Add()
    elif isinstance(stub_layer, StubConcatenate):
        return layers.Concatenate()
    elif isinstance(stub_layer, StubDense):
        return layers.Dense(stub_layer.units, input_shape=(stub_layer.input_units,))

