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
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        name: String. The name of the AutoModel. Defaults to 'image_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        seed: Int. Random seed.
    """

    def __init__(self,
                 num_classes=None,
                 multi_label=False,
                 loss=None,
                 metrics=None,
                 name='image_classifier',
                 max_trials=100,
                 directory=None,
                 seed=None):
        super().__init__(
            outputs=head.ClassificationHead(num_classes=num_classes,
                                            multi_label=multi_label,
                                            loss=loss,
                                            metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            seed=seed)


class ImageRegressor(SupervisedImagePipeline):
    """AutoKeras image regression class.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will infer from the data.
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        name: String. The name of the AutoModel. Defaults to 'image_regressor'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        seed: Int. Random seed.
    """

    def __init__(self,
                 output_dim=None,
                 loss=None,
                 metrics=None,
                 name='image_regressor',
                 max_trials=100,
                 directory=None,
                 seed=None):
        super().__init__(
            outputs=head.RegressionHead(output_dim=output_dim,
                                        loss=loss,
                                        metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            seed=seed)


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
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        name: String. The name of the AutoModel. Defaults to 'text_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
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
                 seed=None):
        super().__init__(
            outputs=head.ClassificationHead(num_classes=num_classes,
                                            multi_label=multi_label,
                                            loss=loss,
                                            metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            seed=seed)


class TextRegressor(SupervisedTextPipeline):
    """AutoKeras text regression class.

    # Arguments
        output_dim: Int. The number of output dimensions. Defaults to None.
            If None, it will infer from the data.
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        name: String. The name of the AutoModel. Defaults to 'text_regressor'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        seed: Int. Random seed.
    """

    def __init__(self,
                 output_dim=None,
                 loss=None,
                 metrics=None,
                 name='text_regressor',
                 max_trials=100,
                 directory=None,
                 seed=None):
        super().__init__(
            outputs=head.RegressionHead(output_dim=output_dim,
                                        loss=loss,
                                        metrics=metrics),
            max_trials=max_trials,
            directory=directory,
            seed=seed)


class SupervisedStructuredDataPipeline(auto_model.AutoModel):

    def __init__(self, outputs, column_names, column_types, **kwargs):
        # TODO: support customized column_types.
        inputs = node.StructuredDataInput()
        if column_types:
            inputs.column_types = column_types
            if column_names is None:
                raise ValueError('Column names must be specified.')
        if column_names:
            inputs.column_names = column_names
        if column_names and column_types:
            for column_name in column_names:
                if column_name not in column_types:
                    raise ValueError('Column_names and column_types are mismatched.')
        super().__init__(inputs=inputs,
                         outputs=outputs,
                         **kwargs)

    def fit(self,
            x=None,  # file path of training data
            y=None,  # label name
            validation_split=0,
            validation_data=None,  # file path of validataion data
            **kwargs):
        """
        # Arguments
            x: numpy.ndarray or tensorflow.Dataset. Training data x.
            If the data is from a csv file, it should be a string specifying the path
            of the csv file of the training data.
            y: numpy.ndarray, tensorflow.Dataset, Training data y.
            If the data is from a csv file, it should be a string corresponding to
            the label column.
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
                If the data is read from a csv file, it should be a string specifying
                the path of a csv file of the validation_data.
            **kwargs: Any arguments supported by keras.Model.fit.
        """
        # x is file path of training data
        if type(x) is str:
            df = pd.read_csv(x)
            validation_df = pd.read_csv(validation_data)
            label = df.pop(y)
            if self.inputs[0].column_names is None:
                column_names = list(df.columns)
                self.inputs[0].column_names = column_names
            else:
                if len(self.inputs[0].column_names) != df.shape[1]:
                    raise ValueError(
                        'The length of column_names and data are mismatched.')
            validation_label = validation_df.pop(y)

            super().fit(x=df,
                        y=label,
                        validation_split=0,
                        validation_data=(validation_df, validation_label),
                        **kwargs)
        # x is numpy ndarray
        else:
            # column_names are not specified:
            if self.inputs[0].column_names is None:
                self.inputs[0].column_names = []
                for index in range(x.shape[1]):
                    self.inputs[0].column_names.append(index)
            super().fit(x=x,
                        y=y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        **kwargs)


class StructuredDataClassifier(SupervisedStructuredDataPipeline):
    """AutoKeras structured data classification class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will infer from the header of the csv file
            or be set to a list of number from zero to the number of columns of the
            data if the data is from a numpy ndarray.
        column_types: Dict. Key is string, indicating the name of a column,
            value is string, indicating the type of that column, should either be
            'numerical' or 'categorical'. The length of the dict should be equal to
            the number of columns of the data. Defaults to None. If not None, the
            column_names need to be specified. If None, it will infer from the data.
        num_classes: Int. Defaults to None. If None, it will infer from the data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to None. If None, the loss will be
            inferred from the AutoModel.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        name: String. The name of the AutoModel. Defaults to
            'structured_data_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
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
            seed=seed)


class StructuredDataRegressor(SupervisedStructuredDataPipeline):
    """AutoKeras structured data regression class.

    # Arguments
        column_names: A list of strings specifying the names of the columns. The
            length of the list should be equal to the number of columns of the data.
            Defaults to None. If None, it will infer from the header of the csv file
            or be set to a list of number from zero to the number of columns of the
            data if the data is from a numpy ndarray.
        column_types: Dict. Key is string, indicating the name of a column,
            value is string, indicating the type of that column, should either be
            'numerical' or 'categorical'. The length of the dict should be equal to
            the number of columns of the data. Defaults to None. If not None, the
            column_names need to be specified. If None, it will infer from the data.
        metrics: A list of Keras metrics. Defaults to None. If None, the metrics will
            be inferred from the AutoModel.
        name: String. The name of the AutoModel. Defaults to
            'structured_data_classifier'.
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trials. Defaults to 100.
        directory: String. The path to a directory for storing the search outputs.
            Defaults to None, which would create a folder with the name of the
            AutoModel in the current directory.
        seed: Int. Random seed.
    """
    def __init__(self,
                 column_names=None,
                 column_types=None,
                 output_dim=None,
                 loss=None,
                 metrics=None,
                 name='structured_data_regressor',
                 max_trials=100,
                 directory=None,
                 seed=None):
        super().__init__(
            outputs=head.RegressionHead(output_dim=output_dim,
                                        loss=loss,
                                        metrics=metrics),
            column_names=column_names,
            column_types=column_types,
            max_trials=max_trials,
            directory=directory,
            seed=seed)
