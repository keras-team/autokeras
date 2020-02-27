import pathlib
from typing import Optional
from typing import Union

from autokeras import auto_model
from autokeras import hypermodels
from autokeras import nodes as input_module
from autokeras import utils
from autokeras.tuners import greedy
from autokeras.tuners import task_specific


class SupervisedTextPipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=input_module.TextInput(),
                         outputs=outputs,
                         **kwargs)


class TextClassifier(SupervisedTextPipeline):
    """AutoKeras text classification class.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        name: String. The name of the AutoModel. Defaults to 'text_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `True`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 num_classes: Optional[int] = None,
                 multi_label: bool = False,
                 loss: utils.AcceptableLoss = None,
                 metrics: utils.AcceptableMetrics = None,
                 name: str = 'text_classifier',
                 max_trials: int = 100,
                 directory: Union[str, pathlib.Path, None] = None,
                 objective: str = 'val_loss',
                 overwrite: bool = True,
                 seed: Optional[int] = None):
        super().__init__(
            outputs=hypermodels.ClassificationHead(num_classes=num_classes,
                                                   multi_label=multi_label,
                                                   loss=loss,
                                                   metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner=task_specific.TextClassifierTuner,
            overwrite=overwrite,
            seed=seed)

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
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The input data
                should be numpy.ndarray or tf.data.Dataset. The data should be one
                dimensional. Each element in the data should be a string which is a
                full sentence.
            y: numpy.ndarray or tensorflow.Dataset. Training data y. It can be raw
                labels, one-hot encoded if more than two classes, or binary encoded
                for binary classification.
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
        super().fit(x=x,
                    y=y,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    **kwargs)


class TextRegressor(SupervisedTextPipeline):
    """AutoKeras text regression class.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will infer from the data.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        name: String. The name of the AutoModel. Defaults to 'text_regressor'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `True`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 output_dim=None,
                 loss='mean_squared_error',
                 metrics=None,
                 name='text_regressor',
                 max_trials=100,
                 directory=None,
                 objective='val_loss',
                 overwrite=True,
                 seed=None):
        super().__init__(
            outputs=hypermodels.RegressionHead(output_dim=output_dim,
                                               loss=loss,
                                               metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner=greedy.Greedy,
            overwrite=overwrite,
            seed=seed)

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
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The input data
                should be numpy.ndarray or tf.data.Dataset. The data should be one
                dimensional. Each element in the data should be a string which is a
                full sentence.
            y: numpy.ndarray or tensorflow.Dataset. Training data y. The targets
                passing to the head would have to be tf.data.Dataset, np.ndarray,
                pd.DataFrame or pd.Series. It can be single-column or multi-column.
                The values should all be numerical.
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
        super().fit(x=x,
                    y=y,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    **kwargs)
