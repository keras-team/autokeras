# Library import
import numpy as np
import tensorflow as tf

import autokeras as ak

# Prepare example Data - Shape 1D
num_instances = 100
num_features = 5
x_train = np.random.rand(num_instances, num_features).astype(np.float32)
y_train = np.zeros(num_instances).astype(np.float32)
y_train[0 : int(num_instances / 2)] = 1
x_test = np.random.rand(num_instances, num_features).astype(np.float32)
y_test = np.zeros(num_instances).astype(np.float32)
y_train[0 : int(num_instances / 2)] = 1

x_train = np.expand_dims(
    x_train, axis=2
)  # This step it's very important an CNN will only accept this data shape
print(x_train.shape)
print(y_train.shape)


# Prepare Automodel for search
input_node = ak.Input()
output_node = ak.ConvBlock()(input_node)
# output_node = ak.DenseBlock()(output_node) #optional
# output_node = ak.SpatialReduction()(output_node) #optional
output_node = ak.ClassificationHead(num_classes=2, multi_label=True)(
    output_node
)

auto_model = ak.AutoModel(
    inputs=input_node, outputs=output_node, overwrite=True, max_trials=1
)


# Search
auto_model.fit(x_train, y_train, epochs=1)
print(auto_model.evaluate(x_test, y_test))


# Export as a Keras Model
model = auto_model.export_model()
print(type(model.summary()))

# print model as image
tf.keras.utils.plot_model(
    model, show_shapes=True, expand_treeed=True, to_file="name.png"
)
