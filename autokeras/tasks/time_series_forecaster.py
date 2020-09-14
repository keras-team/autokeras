# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pandas as pd

from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tasks import structured_data
from autokeras.tuners import greedy
from autokeras.utils import types


class SupervisedTimeseriesDataPipeline(structured_data.BaseStructuredDataPipeline):
    def __init__(
        self,
        outputs,
        column_names=None,
        column_types=None,
        lookback=None,
        predict_from=1,
        predict_until=None,
        **kwargs
    ):
        inputs = input_module.TimeseriesInput(
            lookback=lookback, column_names=column_names, column_types=column_types
        )
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self.predict_from = predict_from
        self.predict_until = predict_until
        self._target_col_name = None
        self.train_len = 0

    @staticmethod
    def _read_from_csv(x, y):
        df = pd.read_csv(x)
        target = df.pop(y).dropna().to_numpy()
        return df, target

    def fit(
        self,
        x=None,
        y=None,
        epochs=None,
        callbacks=None,
        validation_split=0.2,
        validation_data=None,
        **kwargs
    ):
        # x is file path of training data
        if isinstance(x, str):
            self._target_col_name = y
            x, y = self._read_from_csv(x, y)

        if validation_data:
            x_val, y_val = validation_data
            if isinstance(x_val, str):
                validation_data = self._read_from_csv(x_val, y_val)

        self.check_in_fit(x)
        self.train_len = len(y)

        if validation_data:
            x_val, y_val = validation_data
            train_len = len(y_val)
            x_val = x_val[:train_len]
            y_val = y_val[self.lookback - 1 :]
            validation_data = x_val, y_val

        super().fit(
            x=x[: self.train_len],
            y=y[self.lookback - 1 :],
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs
        )

    def predict(self, x, batch_size=32, **kwargs):
        x = self.read_for_predict(x)
        y_pred = super().predict(x=x, batch_size=batch_size, **kwargs)
        lower_bound = self.train_len + self.predict_from
        if self.predict_until is None:
            self.predict_until = len(y_pred)
        upper_bound = min(self.train_len + self.predict_until + 1, len(y_pred))
        return y_pred[lower_bound:upper_bound]

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
        return super().evaluate(
            x=x[: len(y)], y=y[self.lookback - 1 :], batch_size=batch_size, **kwargs
        )


class TimeseriesForecaster(SupervisedTimeseriesDataPipeline):
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
            N + predict_until. If unspecified, it will predict till end of dataset.
            Defaults to None.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        project_name: String. The name of the AutoModel. Defaults to
            'time_series_forecaster'.
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

    def __init__(
        self,
        output_dim=None,
        column_names: Optional[List[str]] = None,
        column_types: Optional[Dict[str, str]] = None,
        lookback: Optional[int] = None,
        predict_from: int = 1,
        predict_until: Optional[int] = None,
        loss: types.LossType = "mean_squared_error",
        metrics: Optional[types.MetricsType] = None,
        project_name: str = "time_series_forecaster",
        max_trials: int = 100,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        tuner: Union[str, Type[tuner.AutoTuner]] = None,
        overwrite: bool = False,
        seed: Optional[int] = None,
        **kwargs
    ):
        if tuner is None:
            tuner = greedy.Greedy
        super().__init__(
            outputs=blocks.RegressionHead(
                output_dim=output_dim, loss=loss, metrics=metrics
            ),
            column_names=column_names,
            column_types=column_types,
            lookback=lookback,
            predict_from=predict_from,
            predict_until=predict_until,
            project_name=project_name,
            max_trials=max_trials,
            directory=directory,
            objective=objective,
            tuner=tuner,
            overwrite=overwrite,
            seed=seed,
            **kwargs
        )
        self.lookback = lookback
        self.predict_from = predict_from
        self.predict_until = predict_until

    def fit(
        self, x=None, y=None, validation_split=0.2, validation_data=None, **kwargs
    ):
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
        super().fit(
            x=x,
            y=y,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs
        )

    def predict(self, x=None, batch_size=32, **kwargs):
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
        return super().predict(x=x, batch_size=batch_size, **kwargs)

    def fit_and_predict(
        self,
        x=None,
        y=None,
        validation_split=0.2,
        validation_data=None,
        batch_size=32,
        **kwargs
    ):
        """Search for the best model and then predict for remaining data points.

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
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        self.fit(
            x=x,
            y=y,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs
        )

        return self.predict(x=x, batch_size=batch_size)


class TimeseriesClassifier(SupervisedTimeseriesDataPipeline):
    """ "AutoKeras time series data classification class.

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
            N + predict_until. If unspecified, it will predict till end of dataset.
            Defaults to None.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        project_name: String. The name of the AutoModel. Defaults to
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
        **kwargs: Any arguments supported by AutoModel.
    """

    def __init__(
        self,
        output_dim=None,
        column_names=None,
        column_types=None,
        lookback=None,
        predict_from=1,
        predict_until=None,
        loss="mean_squared_error",
        metrics=None,
        project_name="time_series_classifier",
        max_trials=100,
        directory=None,
        objective="val_loss",
        overwrite=False,
        seed=None,
        **kwargs
    ):
        raise NotImplementedError

    def fit(
        self, x=None, y=None, validation_split=0.2, validation_data=None, **kwargs
    ):
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
        raise NotImplementedError

    def predict(self, x=None, batch_size=32, **kwargs):
        """Predict the output for a given testing data.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Testing data x, it should also contain the training data used as,
                subsequent predictions depend on them. If the data is from a csv
                file, it should be a string specifying the path of the csv file
                of the testing data.
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.predict.

        # Returns
            A list of numpy.ndarray objects or a single numpy.ndarray.
            The predicted results.
        """
        raise NotImplementedError

    def fit_and_predict(
        self,
        x=None,
        y=None,
        validation_split=0.2,
        validation_data=None,
        batch_size=32,
        **kwargs
    ):
        """Search for the best model and then predict for remaining data points.

        # Arguments
            x: String, numpy.ndarray, pandas.DataFrame or tensorflow.Dataset.
                Training and Test data x. If the data is from a csv file, it
                should be a string specifying the path of the csv file of the
                training data.
            y: String, a list of string(s), numpy.ndarray, pandas.DataFrame or
                tensorflow.Dataset. Training data y.
                If the data is from a csv file, it should be a list of string(s)
                specifying the name(s) of the column(s) need to be forecasted.
                If it is multivariate forecasting, y should be a list of more than
                one column names. If it is univariate forecasting, y should be a
                string or a list of one string.
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
            batch_size: Int. Defaults to 32.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        raise NotImplementedError
