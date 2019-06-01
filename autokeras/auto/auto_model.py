from autokeras.hypermodel.hyper_graph import HyperGraph
from autokeras.layer_utils import format_inputs
from autokeras.tuner import SequentialRandomSearch


class AutoModel(object):
    """ A AutoModel should be an AutoML solution.

    It contains the HyperModels and the Tuner.

    Attributes:
        inputs: A HyperModel instance. The input node of a the AutoModel.
        outputs: A HyperModel instance. The output node of the AutoModel.
        hypermodel: An instance of HyperModelWrap connecting from the inputs to the
            outputs.
        tuner: An instance of Tuner.
    """

    def __init__(self, inputs, outputs, tuner=None):
        """
        """
        self.inputs = inputs
        self.outputs = outputs
        self.hypermodel = None
        self.tuner = tuner
        self.optimizer = None
        self.metrics = None
        self.loss = None

    def hyperparameters(self):
        pass

    def compile(self,
                optimizer=None,
                metrics=None,
                loss=None):
        self.optimizer = optimizer
        self.metrics = metrics
        self.loss = loss

    def fit(self,
            x_train,
            y_train=None,
            tuner=None,
            trails=None,
            single_model_epochs=None):
        # Initialize HyperGraph model
        x_train = format_inputs(x_train, 'train_x')
        y_train = format_inputs(y_train, 'train_y')
        for x in x_train:
            self.inputs.shape = x.shape[1:]
        for y in y_train:
            self.outputs.shape = y.shape[1:]
        self.hypermodel = HyperGraph(self.inputs, self.outputs)

        # Initialize Tuner
        if tuner is not None:
            self.tuner = tuner
        else:
            self.tuner = SequentialRandomSearch(self.hypermodel,
                                                objective=self.metrics),
        self.tuner.search(trails, single_model_epochs=single_model_epochs)

    def predict(self, x_test, postprocessing=True):
        """Predict the output for a given testing data.

        Arguments:
            x_test: The x_test should be a numpy.ndarray.
            postprocessing: Boolean. Mainly for classification task to output
                probabilities instead of labels when set to False.
        """
        pass
