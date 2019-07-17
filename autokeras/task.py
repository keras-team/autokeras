from autokeras import auto_model
from autokeras.hypermodel import head
from autokeras.hypermodel import node


class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=node.ImageInput(),
                         outputs=outputs,
                         **kwargs)


class ImageClassifier(SupervisedImagePipeline):

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(outputs=head.ClassificationHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class ImageRegressor(SupervisedImagePipeline):

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

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(head.ClassificationHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class TextRegressor(SupervisedTextPipeline):

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(head.RegressionHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)
