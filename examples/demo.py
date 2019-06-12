import tensorflow as tf
import autokeras as ak


# Use cases of AutoModel and AutoPipeline

# Simple
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
auto_pipeline = ak.ImageClassifier()

# Loss, optimizer are picked automatically
auto_pipeline.fit(x_train, y_train)

# The predict function should output the labels instead of numerical vectors.
auto_pipeline.predict(x_test)

# Intermediate
inputs = ak.ImageInput(shape=(28, 28, 1))
x = ak.ImageBlock()(inputs)
outputs = ak.ClassificationHead(num_classes=10, metrics=['accuracy'])(x)
automodel = ak.GraphAutoModel(inputs=inputs, outputs=outputs)

# Loss, optimizer are picked automatically
automodel.fit(x_train, y_train)

# Advanced

inputs = ak.ImageInput(shape=(28, 28, 1))
outputs1 = ak.ResNetBlock()(inputs)
outputs2 = ak.XceptionBlock()(inputs)
outputs = ak.Merge()((outputs1, outputs2))

# Even the loss, metrics, num_classes are not provided, they can be inferred.
outputs = ak.ClassificationHead()(outputs)
automodel = ak.GraphAutoModel(inputs=inputs, outputs=outputs)

learning_rate = 1.0

automodel.fit(x_train, y_train,
              trials=100,
              epochs=200,
              callbacks=[tf.keras.callbacks.EarlyStopping(),
                         tf.keras.callbacks.LearningRateScheduler(1)])
