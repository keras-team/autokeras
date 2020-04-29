import kerastuner

from autokeras.engine import serializable


class Preprocessor(kerastuner.HyperModel, serializable.Serializable):
    """Input data preprocessor search space.

    This class defines the search space for input data preprocessor. A
    preprocessor transforms the dataset using `tf.data` operations.
    """

    def build(self, hp, x):
        """Build the `tf.data` input preprocessor.

        # Arguments
            hp: `HyperParameters` instance. The hyperparameters for building the
                model.
            x: `tf.data.Dataset` instance. The input data for preprocessing.

        # Returns
            `tf.data.Dataset`. The preprocessed data to pass to the model.
        """
        raise NotImplementedError
