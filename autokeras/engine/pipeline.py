import kerastuner

from autokeras.engine import serializable


class Pipeline(kerastuner.HyperModel, serializable.Serializable):
    """Input data pipeline search space.

    This class defines the search space for input data pipelines. A pipeline
    transoforms the dataset using tensorflow.data operations.
    """

    def build(self, hp, x, **kwargs):
        """Build the tensorflow.data input pipeline.

        # Returns
            tensorflow.data.Dataset instance.
        """
        raise NotImplementedError
