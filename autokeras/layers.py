from keras import backend
from keras.layers import Add


class WeightedAdd(Add):
    """Weighted Add class inherited from Add class

    It's used to do add weights for data in Add layer

    Attributes:
        weights: backend variable
        one: const 1.0
        kernel: None
        _trainable_weights: list that store weight
    """
    def __init__(self, **kwargs):
        """Init Weighted add class"""
        super(WeightedAdd, self).__init__(**kwargs)
        self.weight = backend.variable(1.0)
        self.one = backend.constant(1.0)
        self.kernel = None
        self._trainable_weights.append(self.weight)

    def call(self, x, **kwargs):
        """Override call function in Add and return new weights"""
        a = backend.tf.scalar_mul(self.weight, x[0])
        b = backend.tf.scalar_mul(backend.tf.subtract(self.one, self.weight), x[1])
        c = backend.tf.add(a, b)
        return c

    def compute_output_shape(self, input_shape):
        """Return output_shape"""
        return input_shape
