from autokeras.auto import auto_model


class AutoPipeline(auto_model.AutoModel):
    """An AutoModel plus preprocessing and postprocessing.

    The preprocessing can include encoding, normalization, and augmentation.
    The postprocessing can include decode the labels from one-hot encoding.
    """

    def fit(self, **kwargs):
        """Tuning the model. """
        pass

    def predict(self, x_test, postprocessing=True):
        """Predict the output for a given testing data.

        Arguments:
            x_test: An instance compatible for input to a Keras Model.
            postprocessing: Boolean. Mainly for classification task to output
                probabilities instead of labels when set to False.

        Returns:
            An instance of numpy.ndarray. It should be labels if
            classification.
        """
        pass

    def build(self, hp):
        raise NotImplementedError
