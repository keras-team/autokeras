from autokeras.auto import auto_model


class AutoPipeline(auto_model.AutoModel):
    """An AutoModel plus preprocessing and postprocessing.

    The preprocessing can include encoding, normalization, and augmentation.
    The postprocessing can include decode the labels from one-hot encoding.
    """

    def build(self, hp, **kwargs):
        raise NotImplementedError
