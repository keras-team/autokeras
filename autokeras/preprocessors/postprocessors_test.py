# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from autokeras import preprocessors
from autokeras.preprocessors import postprocessors


def test_sigmoid_postprocess_to_zero_one():
    postprocessor = postprocessors.SigmoidPostprocessor()

    y = postprocessor.postprocess(np.random.rand(10, 3))

    assert set(y.flatten().tolist()) == set([1, 0])


def test_sigmoid_deserialize_without_error():
    postprocessor = postprocessors.SigmoidPostprocessor()
    dataset = np.array([1, 2])

    postprocessor = preprocessors.deserialize(
        preprocessors.serialize(postprocessor)
    )

    assert isinstance(postprocessor.transform(dataset), np.ndarray)


def test_softmax_postprocess_to_zero_one():
    postprocessor = postprocessors.SoftmaxPostprocessor()

    y = postprocessor.postprocess(np.random.rand(10, 3))

    assert set(y.flatten().tolist()) == set([1, 0])


def test_softmax_transform_dataset_doesnt_change():
    postprocessor = postprocessors.SoftmaxPostprocessor()
    dataset = np.array([1, 2])

    assert isinstance(postprocessor.transform(dataset), np.ndarray)


def test_softmax_deserialize_without_error():
    postprocessor = postprocessors.SoftmaxPostprocessor()
    dataset = np.array([1, 2])

    postprocessor = preprocessors.deserialize(
        preprocessors.serialize(postprocessor)
    )

    assert isinstance(postprocessor.transform(dataset), np.ndarray)


def test_target_normalizer_roundtrip_restores_scale():
    y = np.array([[2.0], [4.0], [6.0], [8.0]], dtype="float32")
    postprocessor = postprocessors.TargetNormalizer()
    postprocessor.fit(y)

    z = postprocessor.transform(y)
    restored = postprocessor.postprocess(z)

    np.testing.assert_allclose(z.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(z.std(), 1.0, atol=1e-5)
    np.testing.assert_allclose(restored, y, atol=1e-5)


def test_target_normalizer_zero_output_decodes_to_mean():
    # A collapsed Dense head that predicts zeros in z-score space must
    # decode to the training mean, not an out-of-scale value (issue #1964).
    y = np.array([[2.0], [4.0], [6.0], [8.0]], dtype="float32")
    postprocessor = postprocessors.TargetNormalizer()
    postprocessor.fit(y)

    decoded = postprocessor.postprocess(np.zeros((4, 1), dtype="float32"))

    np.testing.assert_allclose(decoded, y.mean(), atol=1e-5)


def test_target_normalizer_constant_target_does_not_nan():
    y = np.ones((8, 1), dtype="float32") * 5.2
    postprocessor = postprocessors.TargetNormalizer()
    postprocessor.fit(y)

    z = postprocessor.transform(y)
    restored = postprocessor.postprocess(z)

    assert np.isfinite(z).all()
    np.testing.assert_allclose(restored, y, atol=1e-5)


def test_target_normalizer_deserialize_preserves_stats():
    y = np.array([[1.0], [3.0], [5.0]], dtype="float32")
    postprocessor = postprocessors.TargetNormalizer()
    postprocessor.fit(y)

    restored = preprocessors.deserialize(
        preprocessors.serialize(postprocessor)
    )
    decoded = restored.postprocess(np.zeros((3, 1), dtype="float32"))

    np.testing.assert_allclose(decoded, y.mean(), atol=1e-5)
