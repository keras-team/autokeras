import numpy as np

from autokeras.hypermodel.hyper_graph import HyperGraph
from autokeras.layer_utils import format_inputs, split_train_to_valid
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
            x=None,
            y=None,
            validation_data=None,
            tuner=None,
            trails=None,
            **kwargs):
        # Initialize HyperGraph model
        x = format_inputs(x, 'train_x')
        y = format_inputs(y, 'train_y')
        for x_input, input_node in zip(x, self.inputs):
            input_node.shape = x_input.shape[1:]
        for y_input, output_node in zip(y, self.outputs):
            if len(y_input.shape) == 1:
                y_input = np.reshape(y_input, y_input.shape + (1,))
            output_node.shape = y_input.shape[1:]
        self.hypermodel = HyperGraph(self.inputs, self.outputs)

        # Initialize Tuner
        if tuner is not None:
            self.tuner = tuner
        else:
            self.tuner = SequentialRandomSearch(self.hypermodel,
                                                objective=self.metrics)
        # Prepare the dataset
        if validation_data is None:
            (x, y), (x_val, y_val) = split_train_to_valid(x, y)
            validation_data = x_val, y_val

        # TODO: allow early stop if epochs is not specified.
        self.tuner.search(trails,
                          x=x,
                          y=y,
                          validation_data=validation_data,
                          **kwargs)

    def predict(self, x, **kwargs):
        """Predict the output for a given testing data. """
        return self.tuner.best_model.predict(x, **kwargs)
