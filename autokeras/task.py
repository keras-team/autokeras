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

    def __init__(self, outputs, **kwargs):
        # TODO: support customized column_types.
        super().__init__(inputs=node.StructuredDataInput(),
                         outputs=outputs,
                         **kwargs)

    def fit(self,
            x=None,  # file path of training data
            y=None,  # label name
            validation_split=0,
            validation_data=None,  # file path of validataion data
            **kwargs):
        if type(x) is str:
            df = pd.read_csv(x)
            validation_df = pd.read_csv(validation_data)

            column_names = list(df.columns)
            self.inputs[0].column_names = column_names
            label = df.pop(y)
            validation_label = validation_df.pop(y)
            super().fit(x=df,
                        y=label,
                        validation_split=0,
                        validation_data=(validation_df, validation_label),
                        **kwargs)
        else:
            super().fit(x=x,
                        y=y,
                        validation_split=validation_split,
                        validation_data=validation_data,
                        **kwargs)


class StructuredDataClassifier(SupervisedStructuredDataPipeline):
    """AutoKeras structured data classification class.

    # Arguments
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
            max_trials=max_trials,
            directory=directory,
            seed=seed)


class StructuredDataRegressor(SupervisedStructuredDataPipeline):
    """AutoKeras structured data regression class.

    # Arguments
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
            max_trials=max_trials,
            directory=directory,
            seed=seed)
