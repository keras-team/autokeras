from typing import Callable, Union

import pandas as pd

from autokeras import auto_model
from autokeras.hypermodel import head
from autokeras.hypermodel import node


class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=node.ImageInput(),
                         outputs=outputs,
                         **kwargs)


class ImageClassifier(SupervisedImagePipeline):
    """AutoKeras image classification class.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        name: String. The name of the AutoModel. Defaults to 'image_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 num_classes: int = None,
                 multi_label: bool = False,
                 loss: Union[str, Callable] = None,
                 metrics: Union[str, Callable] = None,
                 name: str = 'image_classifier',
                 max_trials: int = 100,
                 directory: str = None,
                 objective: str = 'val_loss',
                 overwrite=True,
                 seed: int = None):
        super().__init__(
            outputs=head.ClassificationHead(num_classes=num_classes,
                                            multi_label=multi_label,
                                            loss=loss,
                                            metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner='image_classifier',
            overwrite=overwrite,
            seed=seed)

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The shape of the
                data should be 3 or 4 dimensional, the last dimension of which should
                be channel dimension.
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
            validation_split: Float between 0 and 1.
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


class ImageRegressor(SupervisedImagePipeline):
    """AutoKeras image regression class.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        name: String. The name of the AutoModel. Defaults to 'image_regressor'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 output_dim=None,
                 loss='mean_squared_error',
                 metrics=None,
                 name='image_regressor',
                 max_trials=100,
                 directory=None,
                 objective='val_loss',
                 overwrite=True,
                 seed=None):
        super().__init__(
            outputs=head.RegressionHead(output_dim=output_dim,
                                        loss=loss,
                                        metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner='image_regressor',
            overwrite=overwrite,
            seed=seed)

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The shape of the
                data should be 3 or 4 dimensional, the last dimension of which should
                be channel dimension.
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
            validation_split: Float between 0 and 1.
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


class SupervisedTextPipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=node.TextInput(),
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
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 num_classes=None,
                 multi_label=False,
                 loss=None,
                 metrics=None,
                 name='text_classifier',
                 max_trials=100,
                 directory=None,
                 objective='val_loss',
                 overwrite=True,
                 seed=None):
        super().__init__(
            outputs=head.ClassificationHead(num_classes=num_classes,
                                            multi_label=multi_label,
                                            loss=loss,
                                            metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner='text_classifier',
            overwrite=overwrite,
            seed=seed)

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
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
            validation_split: Float between 0 and 1.
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
        multi_label: Boolean. Defaults to False.
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
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
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
            outputs=head.RegressionHead(output_dim=output_dim,
                                        loss=loss,
                                        metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner='text_regressor',
            overwrite=overwrite,
            seed=seed)

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
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
            validation_split: Float between 0 and 1.
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


class SupervisedStructuredDataPipeline(auto_model.AutoModel):

    def __init__(self, outputs, column_names, column_types, **kwargs):
        inputs = node.StructuredDataInput()
        inputs.column_types = column_types
        inputs.column_names = column_names
        if column_types:
            for column_type in column_types.values():
                if column_type not in ['categorical', 'numerical']:
                    raise ValueError(
                        'Column_types should be either "categorical" '
                        'or "numerical", but got {name}'.format(name=column_type))
        if column_names and column_types:
            for column_name in column_types:
                if column_name not in column_names:
                    raise ValueError('Column_names and column_types are '
                                     'mismatched. Cannot find column name '
                                     '{name} in the data.'.format(name=column_name))
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)
        self._target_col_name = None

    def _read_from_csv(self, x, y):
        df = pd.read_csv(x)
        target = df.pop(y).to_numpy()
        return df, target

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
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
            validation_split: Float between 0 and 1.
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
            self._target_column_name = y
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
        if isinstance(x, str):
            x = pd.read_csv(x)
            if self._target_col_name in x:
                x.pop(self._target_col_name)

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
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        num_classes: Int. Defaults to None. If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        name: String. The name of the AutoModel. Defaults to
            'structured_data_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize. Defaults to 'val_accuracy'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 column_names=None,
                 column_types=None,
                 num_classes=None,
                 multi_label=False,
                 loss=None,
                 metrics=None,
                 name='structured_data_classifier',
                 max_trials=100,
                 directory=None,
                 objective='val_accuracy',
                 overwrite=True,
                 seed=None):
        super().__init__(
            outputs=head.ClassificationHead(num_classes=num_classes,
                                            multi_label=multi_label,
                                            loss=loss,
                                            metrics=metrics),
            column_names=column_names,
            column_types=column_types,
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner='structured_data_classifier',
            overwrite=overwrite,
            seed=seed)

    def fit(self,
            x=None,
            y=None,
            epochs=None,
            callbacks=None,
            validation_split=0,
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
            validation_split: Float between 0 and 1.
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
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 column_names=None,
                 column_types=None,
                 output_dim=None,
                 loss='mean_squared_error',
                 metrics=None,
                 name='structured_data_regressor',
                 max_trials=100,
                 directory=None,
                 objective='val_loss',
                 overwrite=True,
                 seed=None):
        super().__init__(
            outputs=head.RegressionHead(output_dim=output_dim,
                                        loss=loss,
                                        metrics=metrics),
            column_names=column_names,
            column_types=column_types,
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner='structured_data_regressor',
            overwrite=overwrite,
            seed=seed)


class TimeSeriesForecaster(auto_model.AutoModel):
    """AutoKeras time series data forecast class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will be obtained from the header of the csv
            file or the pandas.DataFrame.
        column_types: Dict. The keys are the column names. The values should either
            be 'numerical' or 'categorical', indicating the type of that column.
            Defaults to None. If not None, the column_names need to be specified.
            If None, it will be inferred from the data.
        lookback: Int. The range of history steps to consider for each prediction.
            For example, if lookback=n, the data in the range of [i - n, i - 1]
            is used to predict the value of step i. If unspecified, it will be tuned
            automatically.
        predict_from: Int. The starting point of the forecast for each sample (in
            number of steps) after the last time step in the input. If N is the last
            step in the input, then the first step of the predicted output will be
            N + predict_from. Defaults to 1 (which corresponds to starting the
            forecast immediately after the last step in the input).
        predict_until: Int. The end point of the forecast for each sample (in number
            of steps) after the last time step in the input. If N is the last step in
            the input, then the last step of the predicted output will be
            N + predict_until. Defaults to 10.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        name: String. The name of the AutoModel. Defaults to
            'time_series_forecaster'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        objective: String. Name of model metric to minimize
            or maximize, e.g. 'val_accuracy'. Defaults to 'val_loss'.
        overwrite: Boolean. Defaults to `False`. If `False`, reloads an existing
            project of the same name if one is found. Otherwise, overwrites the
            project.
        seed: Int. Random seed.
    """

    def __init__(self,
                 column_names=None,
                 column_types=None,
                 lookback=None,
                 predict_from=1,
                 predict_until=10,
                 loss='mean_squared_error',
                 metrics=None,
                 name='time_series_forecaster',
                 max_trials=100,
                 directory=None,
                 objective='val_loss',
                 overwrite=True,
                 seed=None):
        # TODO: implement.
        raise NotImplementedError

    def fit(self,
            x=None,
            y=None,
            validation_split=0,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the task.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training data x. If the data is from a csv file, it should be a
                string specifying the path of the csv file of the training data.
            y: String, a list of string(s), numpy.ndarray, pandas.DataFrame or
                tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a list of string(s)
                specifying the name(s) of the column(s) need to be forecasted.
                If it is multivariate forecasting, y should be a list of more than
                one column names. If it is univariate forecasting, y should be a
                string or a list of one string.
            validation_split: Float between 0 and 1.
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
        # TODO: implement.
        raise NotImplementedError
