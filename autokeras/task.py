from autokeras import auto_model
from autokeras.hypermodel import hyper_head
from autokeras.hypermodel import hyper_node


class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=hyper_node.ImageInput(),
                         outputs=outputs,
                         **kwargs)


class ImageClassifier(SupervisedImagePipeline):

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(outputs=hyper_head.ClassificationHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class ImageRegressor(SupervisedImagePipeline):

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(hyper_head.RegressionHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class SupervisedTextPipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=hyper_node.TextInput(),
                         outputs=outputs,
                         **kwargs)


class TextClassifier(SupervisedTextPipeline):

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(hyper_head.ClassificationHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)


class TextRegressor(SupervisedTextPipeline):

    def __init__(self, max_trials=None, directory=None, **kwargs):
        super().__init__(hyper_head.RegressionHead(),
                         max_trials=max_trials,
                         directory=directory,
                         **kwargs)
