from autokeras import auto_model
from autokeras.hypermodel import hyper_node
from autokeras.hypermodel import hyper_head


class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=hyper_node.ImageInput(),
                         outputs=outputs,
                         **kwargs)


class ImageClassifier(SupervisedImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(hyper_head.ClassificationHead(), **kwargs)


class ImageRegressor(SupervisedImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(hyper_head.RegressionHead(), **kwargs)


class SupervisedTextPipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=hyper_node.TextInput(),
                         outputs=outputs,
                         **kwargs)


class TextClassifier(SupervisedTextPipeline):
    def __init__(self, **kwargs):
        super().__init__(hyper_head.ClassificationHead(), **kwargs)


class TextRegressor(SupervisedTextPipeline):
    def __init__(self, **kwargs):
        super().__init__(hyper_head.RegressionHead(), **kwargs)
