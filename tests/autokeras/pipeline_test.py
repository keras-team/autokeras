from autokeras import pipeline as pipeline_module
import numpy as np
from autokeras import preprocessors

def test_pipeline_postprocess_one_hot_to_labels():
    pipeline = pipeline_module.Pipeline(inputs=[[]],outputs=[[preprocessors.OneHotEncoder(["a", "b", "c"])]])
    assert np.array_equal(pipeline.postprocess(np.eye(3)), [["a"], ["b"], ["c"]])

def test_pipeline_postprocess_multiple_one_hot_to_labels():
    pipeline = pipeline_module.Pipeline(
        inputs=[[]],
        outputs=[
            [preprocessors.OneHotEncoder(["a", "b", "c"])],
            [preprocessors.OneHotEncoder(["a", "b", "c"])],
        ])
    result = pipeline.postprocess(
        [np.eye(3),
        np.eye(3)]
    )
    assert np.array_equal(result[0], [["a"], ["b"], ["c"]])
    assert np.array_equal(result[1], [["a"], ["b"], ["c"]])
