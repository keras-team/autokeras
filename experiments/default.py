import numpy as np
from keras import Input, Model
from keras.losses import mean_squared_error

from autokeras.layers import ConvBlock, ConvConcat

a1 = Input(shape=(3, 3, 2))
layer = ConvBlock(2)
b = layer(a1)
layer2 = ConvBlock(2)
c = layer2(b)
layer3 = ConvConcat(2)
d = layer3([b, c])
model = Model(inputs=a1, outputs=d)
data = np.random.rand(1, 3, 3, 2)
data2 = np.random.rand(1, 3, 3, 2)
model.compile(optimizer='Adam', loss=mean_squared_error)
w1 = layer.get_weights()
w2 = layer2.get_weights()
w3 = layer3.get_weights()
model.fit(data, data2, epochs=100, verbose=False)
print(w1)
print(layer.get_weights())
print(w2)
print(layer2.get_weights())
print(w3)
print(layer3.get_weights())
