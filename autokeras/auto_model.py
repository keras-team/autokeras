import functools

import kerastuner
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras.hypermodel import processor
from autokeras.hypermodel import hyper_node
from autokeras.hypermodel import hyper_head
from autokeras import utils
from autokeras import tuner
from autokeras import const
from autokeras import meta_model


class AutoModel(kerastuner.HyperModel):
    """ A HyperModel defined by inputs and outputs.

    AutoModel is a subclass of HyperModel. Besides the HyperModel properties,
    it also has a tuner to tune the HyperModel. The user can use it in a similar
    way to a Keras model since it also has `fit()` and  `predict()` methods.

    The user can specify the inputs and outputs of the AutoModel. It will infer
    the rest of the high-level neural architecture.

    Attributes:
        inputs: A list of or a HyperNode instance.
            The input node(s) of the AutoModel.
        outputs: A list of or a HyperHead instance.
            The output head(s) of the AutoModel.
        max_trials: Int. The maximum number of different models to try.
        directory: String. The path to a directory for storing the search outputs.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 max_trials=None,
                 directory=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.tuner = None
        self.max_trials = max_trials or const.Constant.NUM_TRIALS
        self.directory = directory or const.Constant.TEMP_DIRECTORY
        self._node_to_id = {}
        self._nodes = []
        self._hypermodels = []
        self._hypermodel_to_id = {}
        self._model_inputs = []
        self._label_encoders = None
        self._hypermodel_topo_depth = {}
        self._total_topo_depth = 0

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
            outputs = hypermodel.build(hp, inputs=temp_inputs)
            outputs = nest.flatten(outputs)
            for output_node, real_output_node in zip(hypermodel.outputs, outputs):
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
        self._hypermodel_topo_depth = {}
        depth_count = 0
        while len(hypermodels) != 0:
            new_added = []

            # Collect hypermodels with in degree 0.
            for hypermodel in hypermodels:
                if any([in_degree[self._node_to_id[node]]
                        for node in hypermodel.inputs]):
                    continue
                new_added.append(hypermodel)

            # Remove the collected hypermodels from hypermodels.
            for hypermodel in new_added:
                hypermodels.remove(hypermodel)

            for hypermodel in new_added:
                # Add the collected hypermodels to the AutoModel.
                self._add_hypermodel(hypermodel)
                self._hypermodel_topo_depth[
                    self._hypermodel_to_id[hypermodel]] = depth_count

                # Decrease the in degree of the output nodes.
                for output_node in hypermodel.outputs:
                    if output_node not in self._node_to_id:
                        continue
                    output_node_id = self._node_to_id[output_node]
                    in_degree[output_node_id] -= 1

            depth_count += 1
        self._total_topo_depth = depth_count

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

    def _meta_build(self, dataset):
        self.outputs = meta_model.assemble(inputs=self.inputs,
                                           outputs=self.outputs,
                                           dataset=dataset)

        self._build_network()

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
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            y: numpy.ndarray or tensorflow.Dataset. Training data y.
            validation_split: Float between 0 and 1.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
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
        dataset, validation_data = self.prepare_data(
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split)
        self._meta_build(dataset)
        preprocessed_dataset, _ = self.preprocess(
            hp=kerastuner.HyperParameters(),
            dataset=dataset,
            validation_data=validation_data,
            fit=True)
        self.set_node_shapes(preprocessed_dataset)
        self.tuner = tuner.RandomSearch(
            hypermodel=self,
            objective='val_loss',
            max_trials=self.max_trials,
            directory=self.directory)

        # TODO: allow early stop if epochs is not specified.
        self.tuner.search(x=dataset,
                          validation_data=validation_data,
                          **kwargs)

    def set_node_shapes(self, dataset):
        # TODO: Set the shapes only if they are not provided by the user when
        #  initiating the HyperHead or Block.
        x_shapes, y_shapes = utils.dataset_shape(dataset)
        for x_shape, input_node in zip(x_shapes, self._model_inputs):
            input_node.shape = tuple(x_shape.as_list())
        for y_shape, output_node in zip(y_shapes, self.outputs):
            output_node.shape = tuple(y_shape.as_list())
            output_node.in_hypermodels[0].output_shape = output_node.shape

    def prepare_data(self, x, y, validation_data, validation_split):
        # Initialize HyperGraph model
        x = nest.flatten(x)
        y = nest.flatten(y)
        # TODO: check x, y types to be numpy.ndarray or tf.data.Dataset.
        # TODO: y.reshape(-1, 1) if needed.
        y = self._label_encoding(y)
        # Split the data with validation_split
        if (all([isinstance(temp_x, np.ndarray) for temp_x in x]) and
                all([isinstance(temp_y, np.ndarray) for temp_y in y]) and
                validation_data is None and
                validation_split):
            (x, y), (x_val, y_val) = utils.split_train_to_valid(
                x, y,
                validation_split)
            validation_data = x_val, y_val
        # TODO: Handle other types of input, zip dataset, tensor, dict.
        # Prepare the dataset
        dataset = x if isinstance(x, tf.data.Dataset) \
            else utils.prepare_preprocess(x, y)
        if not isinstance(validation_data, tf.data.Dataset):
            x_val, y_val = validation_data
            validation_data = utils.prepare_preprocess(x_val, y_val)
        return dataset, validation_data

    def predict(self, x, batch_size=32, **kwargs):
        """Predict the output for a given testing data. """
        x = utils.prepare_preprocess(x, x)
        x = self.preprocess(self.tuner.get_best_hp(1), x)
        x = x.batch(batch_size)
        y = self.tuner.get_best_models(1)[0].predict(x, **kwargs)
        y = nest.flatten(y)
        y = self._postprocess(y)
        if isinstance(y, list) and len(y) == 1:
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

    def preprocess(self, hp, dataset, validation_data=None, fit=False):
        """Preprocess the data to be ready for the Keras Model.

        Args:
            hp: HyperParameters. Used to build the HyperModel.
            dataset: tf.data.Dataset. Training data.
            validation_data: tf.data.Dataset. Validation data.
            fit: Boolean. Whether to fit the preprocessing layers with x and y.

        Returns:
            if validation data is provided.
            A tuple of two preprocessed tf.data.Dataset, (train, validation).
            Otherwise, return the training dataset.
        """
        for hypermodel in self._hypermodels:
            if isinstance(hypermodel, processor.HyperPreprocessor):
                hypermodel.set_hp(hp)
        dataset = self._preprocess(dataset, fit=fit)
        if not validation_data:
            return dataset
        validation_data = self._preprocess(validation_data)
        return dataset, validation_data

    def _preprocess(self, dataset, fit=False):
        # Put hypermodels in groups by topological depth.
        hypermodels_by_depth = []
        for depth in range(self._total_topo_depth):
            temp_hypermodels = []
            for hypermodel in self._hypermodels:
                if (self._hypermodel_topo_depth[
                    self._hypermodel_to_id[hypermodel]] == depth and
                        isinstance(hypermodel, processor.HyperPreprocessor)):
                    temp_hypermodels.append(hypermodel)
            if not temp_hypermodels:
                break
            hypermodels_by_depth.append(temp_hypermodels)

        # A list of input node ids in the same order as the x in the dataset.
        input_node_ids = [self._node_to_id[input_node] for input_node in self.inputs]

        # Iterate the depth.
        for hypermodels in hypermodels_by_depth:
            if fit:
                # Iterate the dataset to fit the preprocessors in current depth.
                for x, _ in dataset:
                    x = nest.flatten(x)
                    node_id_to_data = {
                        node_id: temp_x
                        for temp_x, node_id in zip(x, input_node_ids)
                    }
                    for hypermodel in hypermodels:
                        data = [node_id_to_data[self._node_to_id[input_node]]
                                for input_node in hypermodel.inputs]
                        hypermodel.update(data)

            for hypermodel in hypermodels:
                hypermodel.finalize()

            # Transform the dataset.
            dataset = dataset.map(functools.partial(
                self._preprocess_transform,
                input_node_ids=input_node_ids,
                hypermodels=hypermodels))

            # Build input_node_ids for next depth.
            input_node_ids = list(sorted([self._node_to_id[hypermodel.outputs[0]]
                                          for hypermodel in hypermodels]))
        return dataset

    def _preprocess_transform(self, x, y, input_node_ids, hypermodels):
        x = nest.flatten(x)
        id_to_data = {
            node_id: temp_x
            for temp_x, node_id in zip(x, input_node_ids)
        }
        output_data = {}
        # Keep the Keras Model inputs even they are not inputs to the hypermodels.
        for node_id, data in id_to_data.items():
            if self._is_model_inputs(self._nodes[node_id]):
                output_data[node_id] = data
        # Transform each x by the corresponding hypermodel
        for hm in hypermodels:
            data = [id_to_data[self._node_to_id[input_node]]
                    for input_node in hm.inputs]
            data = tf.py_function(hm.transform,
                                  inp=nest.flatten(data),
                                  Tout=hm.output_types())
            data = nest.flatten(data)[0]
            data.set_shape(hm.output_shape())
            output_data[self._node_to_id[hm.outputs[0]]] = data
        return tuple(map(
            lambda index: output_data[index], sorted(output_data))), y

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
            head = output_node
            if isinstance(head, hyper_node.Node):
                head = output_node.in_hypermodels[0]
            if (isinstance(head, hyper_head.ClassificationHead) and
                    utils.is_label(temp_y)):
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


class GraphAutoModel(AutoModel):
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

    def __init__(self, *args, **kwargs):
        super(GraphAutoModel, self).__init__(*args, **kwargs)
        self._build_network()

    def _meta_build(self, dataset):
        pass
