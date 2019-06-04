from autokeras import AutoModel


class AutoPipeline(AutoModel):
    """An AutoModel plus preprocessing and postprocessing.

    The preprocessing can include encoding, normalization, and augmentation.
    The postprocessing can include decode the labels from one-hot encoding.
    """

    def fit(self,
            x=None,
            y=None,
            validation_data=None,
            tuner=None,
            trails=None,
            **kwargs):
        """Tuning the model.

        Arguments:
            x: An instance of numpy.ndarray.
            y: An instance of numpy.ndarray.
        """
        pass

    def predict(self, x_test, postprocessing=True):
        """Predict the output for a given testing data.

        Arguments:
            x_test: The x_test should be a numpy.ndarray.
            postprocessing: Boolean. Mainly for classification task to output
                probabilities instead of labels when set to False.

        Returns:
            An instance of numpy.ndarray. It should be labels if
            classification.
        """
        pass