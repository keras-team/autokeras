import kerastuner
import tensorflow as tf
from tensorflow.python.util import nest

from autokeras import meta_model
from autokeras import tuner as tuner_module
from autokeras import utils
from autokeras.hypermodel import base
from autokeras.hypermodel import graph


class AutoModel(object):
    """ A Model defined by inputs and outputs.
    AutoModel combines a HyperModel and a Tuner to tune the HyperModel.
    The user can use it in a similar way to a Keras model since it
    also has `fit()` and  `predict()` methods.

    The AutoModel has two use cases. In the first case, the user only specifies the
    input nodes and output heads of the AutoModel. The AutoModel infers the rest part
    of the model. In the second case, user can specify the high-level architecture of
    the AutoModel by connecting the Blocks with the functional API, which is the same
    as the Keras [functional API](https://www.tensorflow.org/guide/keras/functional).

    # Example
    ```python
        # The user only specifies the input nodes and output heads.
        import autokeras as ak
        ak.AutoModel(
            inputs=[ak.ImageInput(), ak.TextInput()],
            outputs=[ak.ClassificationHead(), ak.RegressionHead()]
        )
    ```
    ```python
        # The user specifies the high-level architecture.
        import autokeras as ak
        image_input = ak.ImageInput()
        image_output = ak.ImageBlock()(image_input)
        text_input = ak.TextInput()
        text_output = ak.TextBlock()(text_input)
        output = ak.Merge()([image_output, text_output])
        classification_output = ak.ClassificationHead()(output)
        regression_output = ak.RegressionHead()(output)
        ak.AutoModel(
            inputs=[image_input, text_input],
            outputs=[classification_output, regression_output]
        )
    ```

    # Arguments
        inputs: A list of Node instances.
            The input node(s) of the AutoModel.
        outputs: A list of Node or Head instances.
            The output node(s) or head(s) of the AutoModel.
        name: String. The name of the AutoModel. Defaults to 'auto_model'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        tuner: String. It should be one of 'greedy', 'bayesian', 'hyperband' or
            'random'. Defaults to 'greedy'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 inputs,
                 outputs,
                 name='auto_model',
                 max_trials=100,
                 directory=None,
                 objective='val_loss',
                 tuner='greedy',
                 overwrite=False,
                 seed=None):
        self.inputs = nest.flatten(inputs)
        self.outputs = nest.flatten(outputs)
        self.name = name
        self.max_trials = max_trials
        self.directory = directory
        self.seed = seed
        self.hyper_graph = None
        self.objective = objective
        # TODO: Support passing a tuner instance.
        self.tuner = tuner_module.get_tuner_class(tuner)
        self.overwrite = overwrite
        self._split_dataset = False
        if all([isinstance(output_node, base.Head)
                for output_node in self.outputs]):
            self.heads = self.outputs
        else:
            self.heads = [output_node.in_blocks[0] for output_node in self.outputs]

    def _meta_build(self, dataset):
        # Using functional API.
        if all([isinstance(output, base.Node) for output in self.outputs]):
            self.hyper_graph = graph.HyperGraph(inputs=self.inputs,
                                                outputs=self.outputs)
        # Using input/output API.
        elif all([isinstance(output, base.Head) for output in self.outputs]):
            self.hyper_graph = meta_model.assemble(inputs=self.inputs,
                                                   outputs=self.outputs,
                                                   dataset=dataset,
                                                   seed=self.seed)
            self.outputs = self.hyper_graph.outputs

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0.2,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            y: numpy.ndarray or tensorflow.Dataset. Training data y.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, by default we train for a maximum of 1000 epochs,
                but we stop training if the validation loss stops improving for 10
                epochs (unless you specified an EarlyStopping callback as part of
                the callbacks argument, in which case the EarlyStopping callback you
                specified will determine early stopping).
            callbacks: List of Keras callbacks to apply during training and
                validation.
            validation_split: Float between 0 and 1. Defaults to 0.2.
                Fraction of the training data to be used as validation data.
                The model will set apart this fraction of the training data,
                will not train on it, and will evaluate
                the loss and any model metrics
                on this data at the end of each epoch.
                The validation data is selected from the last samples
                in the `x` and `y` data provided, before shuffling. This argument is
                not supported when `x` is a dataset.
                The best model found would be fit on the entire dataset including the
                validation data.
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
                The best model found would be fit on the training dataset without the
                validation data.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        dataset, validation_data = self._prepare_data(
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split)

        # Initialize the hyper_graph.
        self._meta_build(dataset)

        # Initialize the Tuner.
        # The hypermodel needs input_shape, which can only be known after
        # preprocessing. So we preprocess the dataset once to get the input_shape,
        # so that the hypermodel can be built in the initializer of the Tuner, which
        # does not access the dataset.
        hp = kerastuner.HyperParameters()
        preprocess_graph, keras_graph = self.hyper_graph.build_graphs(hp)
        preprocess_graph.preprocess(
            dataset=dataset,
            validation_data=validation_data,
            fit=True)
        self.tuner = self.tuner(
            hyper_graph=self.hyper_graph,
            hypermodel=keras_graph,
            fit_on_val_data=self._split_dataset,
            overwrite=self.overwrite,
            objective=self.objective,
            max_trials=self.max_trials,
            directory=self.directory,
            seed=self.seed,
            project_name=self.name)

        # Process the args.
        if callbacks is None:
            callbacks = []
        if epochs is None:
            epochs = 1000
            if not any([isinstance(callback, tf.keras.callbacks.EarlyStopping)
                        for callback in callbacks]):
                callbacks = callbacks + [
                    tf.keras.callbacks.EarlyStopping(patience=10)]

        self.tuner.search(x=dataset,
                          epochs=epochs,
                          callbacks=callbacks,
                          validation_data=validation_data,
                          **kwargs)

    def _process_xy(self, x, y=None, fit=False, predict=False):
        """Convert x, y to tf.data.Dataset.

        # Arguments
            x: Any type allowed by the corresponding input node.
            y: Any type allowed by the corresponding head.
            fit: Boolean. Whether to fit the type converter with the provided data.
            predict: Boolean. If it is called by the predict function of AutoModel.

        # Returns
            A tf.data.Dataset containing both x and y.
        """
        if isinstance(x, tf.data.Dataset):
            if y is None and not predict:
                return x
            if isinstance(y, tf.data.Dataset):
                return tf.data.Dataset.zip((x, y))

        x = nest.flatten(x)
        new_x = []
        for data, input_node in zip(x, self.inputs):
            if fit:
                data = input_node.fit_transform(data)
            else:
                data = input_node.transform(data)
            new_x.append(data)
        x = tf.data.Dataset.zip(tuple(new_x))

        if predict:
            return tf.data.Dataset.zip((x, x))

        if not isinstance(y, tf.data.Dataset):
            y = nest.flatten(y)
            new_y = []
            for data, head_block in zip(y, self.heads):
                if fit:
                    data = head_block.fit_transform(data)
                else:
                    data = head_block.transform(data)
                new_y.append(data)
            y = tf.data.Dataset.zip(tuple(new_y))

        return tf.data.Dataset.zip((x, y))

    def _prepare_data(self, x, y, validation_data, validation_split):
        """Convert the data to tf.data.Dataset."""
        # Check validation information.
        if not validation_data and not validation_split:
            raise ValueError('Either validation_data or validation_split '
                             'should be provided.')
        # TODO: Handle other types of input, zip dataset, tensor, dict.
        # Prepare the dataset.
        dataset = self._process_xy(x, y, fit=True)
        if validation_data:
            self._split_dataset = False
            val_x, val_y = validation_data
            validation_data = self._process_xy(val_x, val_y)
        # Split the data with validation_split.
        if validation_data is None and validation_split:
            self._split_dataset = True
            dataset, validation_data = utils.split_dataset(dataset, validation_split)
        return dataset, validation_data

    def predict(self, x, batch_size=32, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        preprocess_graph, model = self.tuner.get_best_model()
        x = preprocess_graph.preprocess(
            self._process_xy(x, None, predict=True))[0].batch(batch_size)
        y = model.predict(x, **kwargs)
        y = self._postprocess(y)
        if isinstance(y, list) and len(y) == 1:
            y = y[0]
        return y

    def _postprocess(self, y):
        y = nest.flatten(y)
        new_y = []
        for temp_y, head_block in zip(y, self.heads):
            if isinstance(head_block, base.Head):
                temp_y = head_block.postprocess(temp_y)
            new_y.append(temp_y)
        return new_y

    def evaluate(self, x, y=None, batch_size=32, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: Any allowed types according to the input node. Testing data.
            y: Any allowed types according to the head. Testing targets.
                Defaults to None.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        preprocess_graph, model = self.tuner.get_best_model()
        data = preprocess_graph.preprocess(
            self._process_xy(x, y))[0].batch(batch_size)
        return model.evaluate(data, **kwargs)

    def export_model(self):
        """Export the best Keras Model.

        # Returns
            tf.keras.Model instance. The best model found during the search, loaded
            with trained weights.
        """
        _, model = self.tuner.get_best_model()
        return model
