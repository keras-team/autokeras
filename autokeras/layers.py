from keras import backend
from keras.layers import Add


class WeightedAdd(Add):

    def __init__(self, **kwargs):
        super(WeightedAdd, self).__init__(**kwargs)
        self.weight = backend.variable(1.0)
        self.one = backend.constant(1.0)
        self.kernel = None
        self._trainable_weights.append(self.weight)

    def call(self, x, **kwargs):
        a = backend.tf.scalar_mul(self.weight, x[0])
        b = backend.tf.scalar_mul(backend.tf.subtract(self.one, self.weight), x[1])
        c = backend.tf.add(a, b)
        return c

    def compute_output_shape(self, input_shape):
        return input_shape
