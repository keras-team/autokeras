import numpy as np
from keras import Input, Model
from keras.losses import mean_squared_error

from autokeras.layers import ConvBlock

a1 = Input(shape=(3, 3, 2))
layer = ConvBlock(4)
b = layer(a1)
model = Model(inputs=a1, outputs=b)
data = np.random.rand(1, 3, 3, 2)
data2 = np.random.rand(1, 3, 3, 4)
model.compile(optimizer='Adam', loss=mean_squared_error)
print(layer.get_weights())
model.fit(data, data2, epochs=10, verbose=False)
print(layer.get_weights())
