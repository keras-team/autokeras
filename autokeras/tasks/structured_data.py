import pathlib
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd

from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tasks.structured_data_mixin import StructuredDataMixin
from autokeras.tuners import greedy
from autokeras.utils import types


class SupervisedStructuredDataPipeline(StructuredDataMixin, auto_model.AutoModel):

    def __init__(self, outputs, column_names, column_types, **kwargs):
        inputs = input_module.StructuredDataInput()
        inputs.column_types = column_types
        inputs.column_names = column_names
        self.check(column_names, column_types)
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)
        self._target_col_name = None

    @staticmethod
    def _read_from_csv(x, y):
        df = pd.read_csv(x)
        target = df.pop(y).to_numpy()
        return df, target

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0.2,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the task.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the training data.
            y: String, numpy.ndarray, or tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a string, which is the
                name of the target column. Otherwise, it can be single-column or
                multi-column. The values should all be numerical.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, we would use epochs equal to 1000 and early stopping
                with patience equal to 30.
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
        # x is file path of training data
        if isinstance(x, str):
            self._target_col_name = y
            x, y = self._read_from_csv(x, y)
        if validation_data:
            x_val, y_val = validation_data
            if isinstance(x_val, str):
                validation_data = self._read_from_csv(x_val, y_val)

        super().fit(x=x,
                    y=y,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    **kwargs)

    def predict(self, x, batch_size=32, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Testing data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the testing data.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        x = self.read_for_predict(x)

        return super().predict(x=x,
                               batch_size=batch_size,
                               **kwargs)

    def evaluate(self, x, y=None, batch_size=32, **kwargs):
        """Evaluate the best model for the given data.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Testing data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the testing data.
            y: String, numpy.ndarray, or tensorflow.Dataset. Testing data y.
                If the data is from a csv file, it should be a string corresponding
                to the label column.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.evaluate.

        # Returns
            Scalar test loss (if the model has a single output and no metrics) or
            list of scalars (if the model has multiple outputs and/or metrics).
            The attribute model.metrics_names will give you the display labels for
            the scalar outputs.
        """
        if isinstance(x, str):
            x, y = self._read_from_csv(x, y)
        return super().evaluate(x=x,
                                y=y,
                                batch_size=batch_size,
                                **kwargs)


class StructuredDataClassifier(SupervisedStructuredDataPipeline):
    """AutoKeras structured data classification class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data
            excluding the target column. Defaults to None. If None, it will obtained
            from the header of the csv file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        project_name: String. The name of the AutoModel. Defaults to
            'structured_data_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize. Defaults to 'val_accuracy'.
        tuner: String or subclass of AutoTuner. If string, it should be one of
            'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a subclass
            of AutoTuner. If left unspecified, it uses a task specific tuner, which
            first evaluates the most commonly used models for the task before
            exploring other models.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
        **kwargs: Any arguments supported by AutoModel.
    """

    def __init__(self,
                 column_names=None,
                 column_types=None,
                 num_classes=None,
                 multi_label=False,
                 loss=None,
                 metrics=None,
                 project_name='structured_data_classifier',
                 max_trials=100,
                 directory=None,
                 objective='val_accuracy',
                 tuner: Union[str, Type[tuner.AutoTuner]] = None,
                 overwrite=False,
                 seed=None,
                 **kwargs):
        if tuner is None:
            tuner = greedy.Greedy
        super().__init__(
            outputs=blocks.ClassificationHead(num_classes=num_classes,
                                              multi_label=multi_label,
                                              loss=loss,
                                              metrics=metrics),
            column_names=column_names,
            column_types=column_types,
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            objective=objective,
            tuner=tuner,
            overwrite=overwrite,
            seed=seed,
            **kwargs)

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0.2,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the task.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the training data.
            y: String, numpy.ndarray, or tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a string, which is the
                name of the target column. Otherwise, It can be raw labels, one-hot
                encoded if more than two classes, or binary encoded for binary
                classification.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, we would use epochs equal to 1000 and early stopping
                with patience equal to 30.
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
            validation_data: Data on which to evaluate the loss and any model metrics
                at the end of each epoch. The model will not be trained on this data.
                `validation_data` will override `validation_split`. The type of the
                validation data should be the same as the training data.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        super().fit(x=x,
                    y=y,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    **kwargs)


class StructuredDataRegressor(SupervisedStructuredDataPipeline):
    """AutoKeras structured data regression class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data
            excluding the target column. Defaults to None. If None, it will obtained
            from the header of the csv file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will be inferred from the data.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        project_name: String. The name of the AutoModel. Defaults to
            'structured_data_regressor'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        tuner: String or subclass of AutoTuner. If string, it should be one of
            'greedy', 'bayesian', 'hyperband' or 'random'. It can also be a subclass
            of AutoTuner. If left unspecified, it uses a task specific tuner, which
            first evaluates the most commonly used models for the task before
            exploring other models.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
        **kwargs: Any arguments supported by AutoModel.
    """

    def __init__(self,
                 column_names: Optional[List[str]] = None,
                 column_types: Optional[Dict[str, str]] = None,
                 output_dim: Optional[int] = None,
                 loss: types.LossType = 'mean_squared_error',
                 metrics: Optional[types.MetricsType] = None,
                 project_name: str = 'structured_data_regressor',
                 max_trials: int = 100,
                 directory: Union[str, pathlib.Path, None] = None,
                 objective: str = 'val_loss',
                 tuner: Union[str, Type[tuner.AutoTuner]] = None,
                 overwrite: bool = False,
                 seed: Optional[int] = None,
                 **kwargs):
        if tuner is None:
            tuner = greedy.Greedy
        super().__init__(
            outputs=blocks.RegressionHead(output_dim=output_dim,
                                          loss=loss,
                                          metrics=metrics),
            column_names=column_names,
            column_types=column_types,
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            objective=objective,
            tuner=tuner,
            overwrite=overwrite,
            seed=seed,
            **kwargs)
