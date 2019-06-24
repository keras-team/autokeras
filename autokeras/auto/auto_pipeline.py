from autokeras.auto import auto_model


class AutoPipeline(auto_model.AutoModel):
    """An AutoModel plus preprocessing and postprocessing.

    The preprocessing can include encoding, normalization, and augmentation.
    The postprocessing can include decode the labels from one-hot encoding.
    """

    def fit(self, **kwargs):
        """Tuning the model. """
        pass

    def predict(self, x, postprocessing=True):
        """Predict the output for a given testing data.

        Arguments:
            x: Data in a format that can be passed to a Keras model:
                either a Numpy array, a Python generator of arrays, or a TensorFlow Dataset..
            postprocessing: Boolean. Mainly for classification task to output
                probabilities instead of labels when set to False.

        Returns:
            An instance of numpy.ndarray. It should be labels if
            classification.
        """
        pass

    def build(self, hp):
        raise NotImplementedError
