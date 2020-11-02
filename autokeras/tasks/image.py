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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import tensorflow as tf

from autokeras import auto_model
from autokeras import blocks
from autokeras import nodes as input_module
from autokeras.engine import tuner
from autokeras.tuners import greedy
from autokeras.tuners import task_specific
from autokeras.utils import types


class SupervisedImagePipeline(auto_model.AutoModel):
    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=input_module.ImageInput(), outputs=outputs, **kwargs)


class ImageClassifier(SupervisedImagePipeline):
    """AutoKeras image classification class.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        project_name: String. The name of the AutoModel.
            Defaults to 'image_classifier'.
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
        max_model_size: Int. Maximum number of scalars in the parameters of a
            model. Models larger than this are rejected.
        **kwargs: Any arguments supported by AutoModel.
    """

    def __init__(
        self,
        num_classes: Optional[int] = None,
        multi_label: bool = False,
        loss: types.LossType = None,
        metrics: Optional[types.MetricsType] = None,
        project_name: str = "image_classifier",
        max_trials: int = 100,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        tuner: Union[str, Type[tuner.AutoTuner]] = None,
        overwrite: bool = False,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        **kwargs
    ):
        if tuner is None:
            tuner = task_specific.ImageClassifierTuner
        super().__init__(
            outputs=blocks.ClassificationHead(
                num_classes=num_classes,
                multi_label=multi_label,
                loss=loss,
                metrics=metrics,
            ),
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            objective=objective,
            tuner=tuner,
            overwrite=overwrite,
            seed=seed,
            max_model_size=max_model_size,
            **kwargs
        )

    def fit(
        self,
        x: Optional[types.DatasetType] = None,
        y: Optional[types.DatasetType] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        validation_split: Optional[float] = 0.2,
        validation_data: Union[
            tf.data.Dataset, Tuple[types.DatasetType, types.DatasetType], None
        ] = None,
        **kwargs
    ):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The shape of
                the data should be (samples, width, height)
                or (samples, width, height, channels).
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
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        """
        super().fit(
            x=x,
            y=y,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs
        )


class ImageRegressor(SupervisedImagePipeline):
    """AutoKeras image regression class.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will be inferred from the data.
        loss: A Keras loss function. Defaults to use 'mean_squared_error'.
        metrics: A list of Keras metrics. Defaults to use 'mean_squared_error'.
        project_name: String. The name of the AutoModel.
            Defaults to 'image_regressor'.
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
        max_model_size: Int. Maximum number of scalars in the parameters of a
            model. Models larger than this are rejected.
        **kwargs: Any arguments supported by AutoModel.
    """

    def __init__(
        self,
        output_dim: Optional[int] = None,
        loss: types.LossType = "mean_squared_error",
        metrics: Optional[types.MetricsType] = None,
        project_name: str = "image_regressor",
        max_trials: int = 100,
        directory: Union[str, Path, None] = None,
        objective: str = "val_loss",
        tuner: Union[str, Type[tuner.AutoTuner]] = None,
        overwrite: bool = False,
        seed: Optional[int] = None,
        max_model_size: Optional[int] = None,
        **kwargs
    ):
        if tuner is None:
            tuner = greedy.Greedy
        super().__init__(
            outputs=blocks.RegressionHead(
                output_dim=output_dim, loss=loss, metrics=metrics
            ),
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            objective=objective,
            tuner=tuner,
            overwrite=overwrite,
            seed=seed,
            max_model_size=max_model_size,
            **kwargs
        )

    def fit(
        self,
        x: Optional[types.DatasetType] = None,
        y: Optional[types.DatasetType] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        validation_split: Optional[float] = 0.2,
        validation_data: Union[
            types.DatasetType, Tuple[types.DatasetType], None
        ] = None,
        **kwargs
    ):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x. The shape of
                the data should be (samples, width, height) or
                (samples, width, height, channels).
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
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        """
        super().fit(
            x=x,
            y=y,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs
        )


class ImageSegmenter(SupervisedImagePipeline):
    """AutoKeras image segmentation class.
    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        loss: A Keras loss function. Defaults to use 'binary_crossentropy' or
            'categorical_crossentropy' based on the number of classes.
        metrics: A list of metrics used to measure the accuracy of the model,
            default to 'accuracy'.
        project_name: String. The name of the AutoModel.
            Defaults to 'image_segmenter'.
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
        num_classes: Optional[int] = None,
        loss: types.LossType = None,
        metrics: Optional[types.MetricsType] = None,
        project_name: str = "image_segmenter",
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
            outputs=blocks.SegmentationHead(
                num_classes=num_classes, loss=loss, metrics=metrics
            ),
            max_trials=max_trials,
            directory=directory,
            project_name=project_name,
            objective=objective,
            tuner=tuner,
            overwrite=overwrite,
            seed=seed,
            **kwargs
        )

    def fit(
        self,
        x: Optional[types.DatasetType] = None,
        y: Optional[types.DatasetType] = None,
        epochs: Optional[int] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None,
        validation_split: Optional[float] = 0.2,
        validation_data: Union[
            types.DatasetType, Tuple[types.DatasetType], None
        ] = None,
        **kwargs
    ):
        """Search for the best model and hyperparameters for the AutoModel.

        It will search for the best model based on the performances on
        validation data.

        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training image dataset x.
                The shape of the data should be (samples, width, height) or
                (samples, width, height, channels).
            y: numpy.ndarray or tensorflow.Dataset. Training image data set y.
                It should be a tensor and the height and width should be the same
                as x. Each element in the tensor is the label of the corresponding
                pixel.
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
            **kwargs: Any arguments supported by
                [keras.Model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
        """
        super().fit(
            x=x,
            y=y,
            epochs=epochs,
            callbacks=callbacks,
            validation_split=validation_split,
            validation_data=validation_data,
            **kwargs
        )
