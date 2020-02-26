from autokeras import auto_model
from autokeras import nodes as input_module
from autokeras import hypermodels
# from keras import
from autokeras.tuners import task_specific
class SupervisedSegmentorPipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=input_module.ImageInput(),
                         outputs=outputs,
                         **kwargs)
class TimeSeriesForecaster(SupervisedSegmentorPipeline):
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
        overwrite: Boolean. Defaults to `True`. If `False`, reloads an existing
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
        super().__init__(
            outputs=hypermodels.SegmentationHead(column_names=column_names,
                                                 column_types=column_types,
                                                 lookback = lookback,
                                                 predict_from = predict_from,
                                                 predict_until = predict_until,
                                                   loss=loss,
                                                   metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            name=name,
            objective=objective,
            tuner=task_specific.ImageSetmenterTuner,
            overwrite=overwrite,
            seed=seed)
        # TODO: implement.
        raise NotImplementedError

    def fit(self,
            x=None,
            y=None,
            epochs = None,
            callbacks=None,
            validation_split=0.2,
            validation_data=None,
            **kwargs):
        """Search for the best model and hyperparameters for the task.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset.
                Training data x. The shape of the data should be 3 or 4 dimensional,
                the last dimension of which should be channel dimension.
                The dataset is the image dataset.
            y: numpy.ndarray or tensorflow.Dataset.
                Training data y. The shape of the data should be 3 or 4 dimensional,
                the last dimension of which should be channel dimension.
                The dataset should be the ground truth image dataset.
            epochs: Int. The number of epochs to train each model during the search.
                If unspecified, by default we train for a maximum of 1000 epochs,
                but we stop training if the validation loss stops improving for 10
                epochs (unless you specified an EarlyStopping callback as part of
                the callbacks argument, in which case the EarlyStopping callback you
                specified will determine early stopping).
            callbacks: List of Keras callbacks to apply during training and
                validation.
                The dataset should be the ground truth segmentation image dataset.
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
        # TODO: implement.
        dataset, validation_data = self._prepare_data(
            x=x,
            y=y,
            validation_data=validation_data,
            validation_split=validation_split)
        super().fit(x=x,
                    y=y,
                    epochs=epochs,
                    callbacks=callbacks,
                    validation_split=validation_split,
                    validation_data=validation_data,
                    **kwargs)
        raise NotImplementedError
