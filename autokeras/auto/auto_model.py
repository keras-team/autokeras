import numpy as np

from autokeras.dataset import Dataset
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
        self.inputs = format_inputs(inputs)
        self.outputs = format_inputs(outputs)
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
            x_train=None,
            y_train=None,
            x_valid=None,
            y_valid=None,
            tuner=None,
            trails=None,
            **kwargs):
        # Initialize HyperGraph model
        x_train = format_inputs(x_train, 'train_x')
        y_train = format_inputs(y_train, 'train_y')
        for x, input_node in zip(x_train, self.inputs):
            input_node.shape = x.shape[1:]
        for y, output_node in zip(y_train, self.outputs):
            if len(y.shape) == 1:
                y = np.reshape(y, y.shape + (1,))
            output_node.shape = y.shape[1:]
        self.hypermodel = HyperGraph(self.inputs, self.outputs)

        # Initialize Tuner
        if tuner is not None:
            self.tuner = tuner
        else:
            self.tuner = SequentialRandomSearch(self.hypermodel,
                                                objective=self.metrics)
        # Prepare the dataset
        dataset = Dataset(x_train=x_train,
                          y_train=y_train,
                          x_valid=x_valid,
                          y_valid=y_valid)
        if any([x_train, y_train]) and not any([x_valid, y_valid]):
            dataset.split_train_to_valid()

        # TODO: allow early stop if epochs is not specified.
        self.tuner.search(trails,
                          x=x_train,
                          y=y_train,
                          validation_data=(dataset.x_valid, dataset.y_valid),
                          **kwargs)

    def predict(self, x, **kwargs):
        """Predict the output for a given testing data. """
        return self.tuner.best_model.predict(x, **kwargs)
