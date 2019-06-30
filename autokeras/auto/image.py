from autokeras.auto import auto_model
from autokeras.hypermodel import hyper_node, processor
from autokeras.hypermodel import hyper_head


class SupervisedImagePipeline(auto_model.AutoModel):

    def __init__(self, outputs, **kwargs):
        super().__init__(inputs=hyper_node.ImageInput(),
                         outputs=outputs,
                         **kwargs)


class ImageClassifier(SupervisedImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(hyper_head.ClassificationHead(), **kwargs)
        self.label_encoder = processor.OneHotEncoder()

    def fit(self, x=None, y=None, **kwargs):
        self.label_encoder.fit(y)
        super().fit(x=x, y=self.label_encoder.transform(y), **kwargs)

    def predict(self, x, **kwargs):
        return self.label_encoder.inverse_transform(super().predict(x, **kwargs))


class ImageRegressor(SupervisedImagePipeline):
    def __init__(self, **kwargs):
        super().__init__(hyper_head.RegressionHead(), **kwargs)
