
class AutoModel:
    """ A AutoModel should be an AutoML solution.

    It contains the HyperModels and the Tuner.

    Attributes:
        inputs: A HyperModel instance. The input node of a the AutoModel.
        outputs: A HyperModel instance. The output node of the AutoModel.
        hypermodels: A list of HyperModels connecting from the inputs to the
            outputs.
        tuner: An instance of Tuner.
    """
    def __init__(self, inputs, outputs, tuner=None):
        """
        """
        self.inputs = inputs
        self.outputs = outputs
        self.hypermodels = []
        self.tuner = tuner

    def hyperparameters(self):
        pass

    def compile(self, optimizer=None, metrics=None, loss=None):
        pass

    def fit(self, x_train, y_train=None, **kwargs):
        pass

    def predict(self, x_test, postprocessing=True):
        """Predict the output for a given testing data.

        Arguments:
            x_test: The x_test should be a numpy.ndarray.
            postprocessing: Boolean. Mainly for classification task to output
                probabilities instead of labels when set to False.
        """
        pass


