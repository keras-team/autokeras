from autokeras import hyper_preprocessors
from autokeras import preprocessors

def test_serialize_and_deserialize_default_hpps():
    preprocessor = preprocessors.AddOneDimension()
    hyper_preprocessor = hyper_preprocessors.DefaultHyperPreprocessor(preprocessor)
    hyper_preprocessor = hyper_preprocessors.deserialize(hyper_preprocessors.serialize(hyper_preprocessor))
    assert isinstance(hyper_preprocessor.preprocessor, preprocessors.AddOneDimension)
