import kerastuner

from autokeras.engine import serializable


class Preprocessor(kerastuner.HyperModel, serializable.Serializable):
    """Input data preprocessor search space.

    This class defines the search space for input data preprocessor. A
    preprocessor transoforms the dataset using tensorflow.data operations.
    """

    def build(self, hp, x):
        """Build the tensorflow.data input preprocessor.

        # Arguments
            hp: HyperParameters. The hyperparameters for building the model.
            x: An instance of tensorflow.data.Dataset.

        # Returns
            tensorflow.data.Dataset instance.
        """
        raise NotImplementedError
