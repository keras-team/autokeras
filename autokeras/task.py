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

    Attributes:
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trails. Defaults to 100.
        directory: Str. Path to a directory for storing temporary files during
            the search. Defaults to None.
    """

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(outputs=head.ClassificationHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class ImageRegressor(SupervisedImagePipeline):
    """AutoKeras image regression class.

    Attributes:
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trails. Defaults to 100.
        directory: Str. Path to a directory for storing temporary files during
            the search. Defaults to None.
    """

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(head.RegressionHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class SupervisedTextPipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=node.TextInput(),
                         outputs=outputs,
                         **kwargs)


class TextClassifier(SupervisedTextPipeline):
    """AutoKeras text classification class.

    Attributes:
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trails. Defaults to 100.
        directory: Str. Path to a directory for storing temporary files during
            the search. Defaults to None.
    """

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(head.ClassificationHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class TextRegressor(SupervisedTextPipeline):
    """AutoKeras text regression class.

    Attributes:
        max_trials: Int. The maximum number of different Keras Models to try.
            The search may finish before reaching the max_trails. Defaults to 100.
        directory: Str. Path to a directory for storing temporary files during
            the search. Defaults to None.
    """

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(head.RegressionHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)
