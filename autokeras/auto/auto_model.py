import queue
import kerastuner
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras.auto import tuner
from autokeras.hypermodel import hyper_block, processor
from autokeras.hypermodel import hyper_head
from autokeras import layer_utils
from autokeras import const


class GraphAutoModel(kerastuner.HyperModel):
    """A HyperModel defined by a graph of HyperBlocks.

    GraphAutoModel is a subclass of HyperModel. Besides the HyperModel properties,
    it also has a tuner to tune the HyperModel. The user can use it in a similar
    way to a Keras model since it also has `fit()` and  `predict()` methods.

    The user can specify the high-level neural architecture by connecting the
    HyperBlocks with the functional API, which is the same as
    the Keras functional API.

    Attributes:
        inputs: A list of or a HyperNode instances.
            The input node(s) of the GraphAutoModel.
        outputs: A list of or a HyperNode instances.
            The output node(s) of the GraphAutoModel.
        max_trials: Int. The maximum number of different models to try.
        directory: String. The path to the directory
            for storing the search outputs.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 max_trials=None,
                 directory=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.inputs = layer_utils.format_inputs(inputs)
        self.outputs = layer_utils.format_inputs(outputs)
        self.tuner = None
        self.max_trials = max_trials or const.Constant.NUM_TRIALS
        self.directory = directory or const.Constant.TEMP_DIRECTORY
        self._node_to_id = {}
        self._nodes = []
        self._hypermodels = []
        self._hypermodel_to_id = {}
        self._model_inputs = []
        self._build_network()
        self._label_encoders = None

    def build(self, hp):
        real_nodes = {}
        for input_node in self._model_inputs:
            node_id = self._node_to_id[input_node]
            real_nodes[node_id] = input_node.build()
        for hypermodel in self._hypermodels:
            if isinstance(hypermodel, processor.HyperPreprocessor):
                continue
            temp_inputs = [real_nodes[self._node_to_id[input_node]]
                           for input_node in hypermodel.inputs]
            outputs = hypermodel.build(hp,
                                       inputs=temp_inputs)
            outputs = layer_utils.format_inputs(outputs, hypermodel.name)
            for output_node, real_output_node in zip(hypermodel.outputs,
                                                     outputs):
                real_nodes[self._node_to_id[output_node]] = real_output_node
        model = tf.keras.Model(
            [real_nodes[self._node_to_id[input_node]] for input_node in
             self._model_inputs],
            [real_nodes[self._node_to_id[output_node]] for output_node in
             self.outputs])

        return self._compiled(hp, model)

    def _build_network(self):
        self._node_to_id = {}

        # Recursively find all the interested nodes.
        for input_node in self.inputs:
            self._search_network(input_node, self.outputs, set(), set())
        self._nodes = sorted(list(self._node_to_id.keys()),
                             key=lambda x: self._node_to_id[x])

        for node in (self.inputs + self.outputs):
            if node not in self._node_to_id:
                raise ValueError('Inputs and outputs not connected.')

        # Find the hypermodels.
        hypermodels = set()
        for input_node in self._nodes:
            for hypermodel in input_node.out_hypermodels:
                if any([output_node in self._node_to_id
                        for output_node in hypermodel.outputs]):
                    hypermodels.add(hypermodel)

        # Check if all the inputs of the hypermodels are set as inputs.
        for hypermodel in hypermodels:
            for input_node in hypermodel.inputs:
                if input_node not in self._node_to_id:
                    raise ValueError('A required input is missing for HyperModel '
                                     '{name}.'.format(name=hypermodel.name))

        # Calculate the in degree of all the nodes
        in_degree = [0] * len(self._nodes)
        for node_id, node in enumerate(self._nodes):
            in_degree[node_id] = len([
                hypermodel
                for hypermodel in node.in_hypermodels if hypermodel in hypermodels
            ])

        # Add the hypermodels in topological order.
        self._hypermodels = []
        self._hypermodel_to_id = {}
        while len(hypermodels) != 0:
            new_added = []
            for hypermodel in hypermodels:
                if any([in_degree[self._node_to_id[node]]
                        for node in hypermodel.inputs]):
                    continue
                self._add_hypermodel(hypermodel)
                new_added.append(hypermodel)
                for output_node in hypermodel.outputs:
                    if output_node not in self._node_to_id:
                        continue
                    output_node_id = self._node_to_id[output_node]
                    in_degree[output_node_id] -= 1
            for hypermodel in new_added:
                hypermodels.remove(hypermodel)

        # Set the output shape.
        for output_node in self.outputs:
            hypermodel = output_node.in_hypermodels[0]
            hypermodel.output_shape = output_node.shape

        # Find the model input nodes
        for node in self._nodes:
            if self._is_model_inputs(node):
                self._model_inputs.append(node)

        self._model_inputs = sorted(self._model_inputs,
                                    key=lambda x: self._node_to_id[x])

    def _search_network(self, input_node, outputs, in_stack_nodes,
                        visited_nodes):
        visited_nodes.add(input_node)
        in_stack_nodes.add(input_node)

        outputs_reached = False
        if input_node in outputs:
            outputs_reached = True

        for hypermodel in input_node.out_hypermodels:
            for output_node in hypermodel.outputs:
                if output_node in in_stack_nodes:
                    raise ValueError('The network has a cycle.')
                if output_node not in visited_nodes:
                    self._search_network(output_node, outputs, in_stack_nodes,
                                         visited_nodes)
                if output_node in self._node_to_id.keys():
                    outputs_reached = True

        if outputs_reached:
            self._add_node(input_node)

        in_stack_nodes.remove(input_node)

    def _add_hypermodel(self, hypermodel):
        if hypermodel not in self._hypermodels:
            hypermodel_id = len(self._hypermodels)
            self._hypermodel_to_id[hypermodel] = hypermodel_id
            self._hypermodels.append(hypermodel)

    def _add_node(self, input_node):
        if input_node not in self._node_to_id:
            self._node_to_id[input_node] = len(self._node_to_id)

    def fit(self,
            x=None,
            y=None,
            validation_split=0,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        Args:
            x: Any type compatible with Keras training x. Training data x.
            y: Any type compatible with Keras training y. Training data y.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset, dataset iterator, generator or
               `keras.utils.Sequence` instance.
            validation_data: Data on which to evaluate
                the loss and any model metrics at the end of each epoch.
                The model will not be trained on this data.
                `validation_data` will override `validation_split`.
                `validation_data` could be:
                  - tuple `(x_val, y_val)` of Numpy arrays or tensors
                  - tuple `(x_val, y_val, val_sample_weights)` of Numpy arrays
                  - dataset or a dataset iterator
                For the first two cases, `batch_size` must be provided.
                For the last case, `validation_steps` must be provided.
        """
        # Initialize HyperGraph model
        x = layer_utils.format_inputs(x, 'train_x')
        y = layer_utils.format_inputs(y, 'train_y')

        y = self._label_encoding(y)
        # TODO: Set the shapes only if they are not provided by the user when
        #  initiating the HyperHead or Block.
        for y_input, output_node in zip(y, self.outputs):
            if len(y_input.shape) == 1:
                y_input = np.reshape(y_input, y_input.shape + (1,))
            output_node.shape = y_input.shape[1:]
            output_node.in_hypermodels[0].output_shape = output_node.shape

        # Prepare the dataset
        if (all([isinstance(temp_x, np.ndarray) for temp_x in x]) and
                validation_data is None and
                validation_split):
            (x, y), (x_val, y_val) = layer_utils.split_train_to_valid(
                x, y,
                validation_split)
            validation_data = x_val, y_val
            validation_split = 0

        self.preprocess(hp=kerastuner.HyperParameters(),
                        x=x,
                        y=y,
                        validation_data=validation_data)
        self.tuner = tuner.RandomSearch(
            hypermodel=self,
            objective='val_loss',
            max_trials=self.max_trials,
            directory=self.directory)

        # TODO: allow early stop if epochs is not specified.
        self.tuner.search(x=x,
                          y=y,
                          validation_split=validation_split,
                          validation_data=validation_data,
                          **kwargs)

    def predict(self, x, **kwargs):
        """Predict the output for a given testing data. """
        x, _, _ = self.preprocess(self.tuner.get_best_models(1), x)
        y = self.tuner.get_best_models(1)[0].predict(x, **kwargs)
        y = layer_utils.format_inputs(y, self.name)
        y = self._postprocess(y)
        if len(y) == 1:
            y = y[0]
        return y

    def _get_metrics(self):
        metrics = []
        for metrics_list in [output_node.in_hypermodels[0].metrics for
                             output_node in self.outputs
                             if isinstance(output_node.in_hypermodels[0],
                                           hyper_head.HyperHead)]:
            metrics += metrics_list
        return metrics

    def _get_loss(self):
        loss = nest.flatten([output_node.in_hypermodels[0].loss
                             for output_node in self.outputs
                             if isinstance(output_node.in_hypermodels[0],
                                           hyper_head.HyperHead)])
        return loss

    def _compiled(self, hp, model):
        # Specify hyperparameters from compile(...)
        optimizer = hp.Choice('optimizer',
                              ['adam',
                               'adadelta',
                               'sgd'])

        model.compile(optimizer=optimizer,
                      metrics=self._get_metrics(),
                      loss=self._get_loss())

        return model

    def preprocess(self, hp, x, y=None, validation_data=None):
        """Preprocess the data to be ready for the Keras Model.

        Args:
            hp: HyperParameters. Used to build the HyperModel.
            x: Any type compatible with Keras training x. Training data x.
            y: Any type compatible with Keras training y. Training data y.
            validation_data: Tuple of (val_x, val_y). The same type as x and y.
                Validation set for the search.

        Returns:
            A tuple of four preprocessed elements, (x, y, val_x, val_y).

        """
        x, y = self._preprocess(hp, x, y)
        if not y:
            return x, None, None
        if not validation_data:
            return x, y, None
        val_x, val_y = validation_data
        val_x, val_y = self._preprocess(hp, val_x, val_y)
        return x, y, (val_x, val_y)

    def _preprocess(self, hp, x, y):
        x = layer_utils.format_inputs(x, self.name)
        q = queue.Queue()
        for input_node, data in zip(self.inputs, x):
            q.put((input_node, data))

        new_x = []
        while not q.empty():
            node, data = q.get()
            if self._is_model_inputs(node):
                new_x.append((self._node_to_id[node], data))
                node.shape = data.shape[1:]

            for hypermodel in node.out_hypermodels:
                if isinstance(hypermodel, processor.HyperPreprocessor):
                    q.put((hypermodel.outputs[0],
                           hypermodel.fit_transform(hp, data)))
        # Sort by id.
        new_x = sorted(new_x, key=lambda a: a[0])

        # Remove the id from the list.
        return_x = []
        for node_id, data in new_x:
            self._nodes[node_id].shape = data.shape[1:]
            return_x.append(data)
        return return_x, y

    @staticmethod
    def _is_model_inputs(node):
        for hypermodel in node.in_hypermodels:
            if not isinstance(hypermodel, processor.HyperPreprocessor):
                return False
        for hypermodel in node.out_hypermodels:
            if not isinstance(hypermodel, processor.HyperPreprocessor):
                return True
        return False

    def _label_encoding(self, y):
        self._label_encoders = []
        new_y = []
        for temp_y, output_node in zip(y, self.outputs):
            head = output_node.in_hypermodels[0]
            if (isinstance(head, hyper_head.ClassificationHead) and
                    self._is_label(temp_y)):
                label_encoder = processor.OneHotEncoder()
                label_encoder.fit(y)
                new_y.append(label_encoder.transform(y))
                self._label_encoders.append(label_encoder)
            else:
                new_y.append(temp_y)
                self._label_encoders.append(None)
        return new_y

    def _postprocess(self, y):
        new_y = []
        for temp_y, label_encoder in zip(y, self._label_encoders):
            if label_encoder:
                new_y.append(label_encoder.inverse_transform(temp_y))
            else:
                new_y.append(temp_y)
        return new_y

    @staticmethod
    def _is_label(y):
        return len(y.flatten()) == len(y) and len(set(y.flatten())) > 2


class AutoModel(GraphAutoModel):
    """ A HyperModel defined by inputs and outputs.

    AutoModel is a subclass of HyperModel. Besides the HyperModel properties,
    it also has a tuner to tune the HyperModel. The user can use it in a similar
    way to a Keras model since it also has `fit()` and  `predict()` methods.

    The user can specify the inputs and outputs of the AutoModel. It will infer
    the rest of the high-level neural architecture.

    Attributes:
        inputs: A list of or a HyperNode instance.
            The input node(s) of a the AutoModel.
        outputs: A list of or a HyperHead instance.
            The output head(s) of the AutoModel.
        max_trials: Int. The maximum number of different models to try.
        directory: String. The path to a directory for storing the search outputs.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 **kwargs):
        inputs = layer_utils.format_inputs(inputs)
        outputs = layer_utils.format_inputs(outputs)
        middle_nodes = [input_node.related_block()(input_node)
                        for input_node in inputs]
        if len(middle_nodes) > 1:
            output_node = hyper_block.Merge()(middle_nodes)
        else:
            output_node = middle_nodes[0]
        outputs = [output_blocks(output_node)
                   for output_blocks in outputs]
        super().__init__(inputs, outputs, **kwargs)
