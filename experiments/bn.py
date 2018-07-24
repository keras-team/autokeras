import numpy as np

from keras import Sequential
from keras.layers import BatchNormalization

bn = BatchNormalization(input_shape=(28, 28, 3), epsilon=0)
model = Sequential([bn])
model.compile(optimizer='rmsprop',
              loss='mse')

n_filters = 3
new_weights = [
    np.ones(n_filters, dtype=np.float32),
    np.zeros(n_filters, dtype=np.float32),
    np.zeros(n_filters, dtype=np.float32),
    np.ones(n_filters, dtype=np.float32)
]
bn.set_weights(new_weights)

x_train = np.random.rand(2, 28, 28, 3)
output = model.predict_on_batch(x_train)

print(x_train.shape)
print(output.shape)
print(np.sum(np.abs(x_train - output)))
