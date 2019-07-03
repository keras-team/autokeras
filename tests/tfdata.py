import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.data import Dataset

inputs1 = keras.Input(shape=(32,), name='a')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs1)
x1 = layers.Dense(64, activation='relu', name='dense_2')(x)
inputs2 = keras.Input(shape=(32,), name='b')
x = layers.Dense(64, activation='relu', name='dense_3')(inputs2)
x2 = layers.Dense(64, activation='relu', name='dense_4')(x)
x = layers.Add()([x1, x2])
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)

model = keras.Model(inputs=[inputs1, inputs2], outputs=outputs)
model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

def random_one_hot_labels(shape):
  n, n_class = shape
  classes = np.random.randint(0, n_class, n)
  labels = np.zeros((n, n_class))
  labels[np.arange(n), classes] = 1
  return labels

data = np.random.random((1000, 32))
labels = random_one_hot_labels((1000, 10))
val_data = np.random.random((100, 32))
val_labels = random_one_hot_labels((100, 10))

# dataset = tf.data.Dataset.from_tensor_slices(({'a': data, 'b': data}, labels))
dataset = tf.data.Dataset.from_tensor_slices(((data, data), labels))
dataset = dataset.batch(32)

val_dataset = tf.data.Dataset.from_tensor_slices(({'a': val_data, 'b': val_data}, val_labels))
val_dataset = val_dataset.batch(32)

model.fit(dataset, epochs=10, steps_per_epoch=30,
          validation_data=val_dataset,
          validation_steps=3)

model.evaluate(dataset)
test_dataset = tf.data.Dataset.from_tensor_slices(((data, data), data))
test_dataset = test_dataset.batch(32)
model.predict(test_dataset)


a = Dataset.range(1, 4)  # ==> [ 1, 2, 3 ]
b = Dataset.range(4, 7)  # ==> [ 4, 5, 6 ]
c = Dataset.range(7, 13).batch(2)  # ==> [ [7, 8], [9, 10], [11, 12] ]
d = Dataset.range(13, 15)  # ==> [ 13, 14 ]

# The nested structure of the `datasets` argument determines the
# structure of elements in the resulting dataset.
Dataset.zip((a, b))  # ==> [ (1, 4), (2, 5), (3, 6) ]
Dataset.zip((b, a))  # ==> [ (4, 1), (5, 2), (6, 3) ]

# The `datasets` argument may contain an arbitrary number of
# datasets.
Dataset.zip((a, b, c))  # ==> [ (1, 4, [7, 8]),
                        #       (2, 5, [9, 10]),
                        #       (3, 6, [11, 12]) ]

# The number of elements in the resulting dataset is the same as
# the size of the smallest dataset in `datasets`.
Dataset.zip((a, d))  # ==> [ (1, 13), (2, 14) ]
print(Dataset.zip((tuple([a, b]), d)))  # ==> [ (1, 13), (2, 14) ]

