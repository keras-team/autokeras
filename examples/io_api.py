import autokeras as ak
from keras.datasets import mnist

# Prepare the data.
(x_train, y_classification), (x_test, y_test) = mnist.load_data()
x_image = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

y_regression = np.random()

# Build model and train.
automodel = ak.AutoModel(
   inputs=[ak.ImageInput(),
           ak.StructuredDataInput()],
   outputs=[ak.RegressionHead(metrics=['mae']),
            ak.ClassificationHead(loss='categorical_crossentropy',
                                  metrics=['accuracy'])])
automodel.fit([x_image, x_structured],
              [y_regression, y_classification])
