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
import pandas as pd

import autokeras as ak
from autokeras import test_utils


def test_text_and_structured_data(tmp_path):
    # Prepare the data.
    num_instances = 3
    x_text = test_utils.generate_text_data(num_instances)

    x_structured_data = pd.read_csv(test_utils.TRAIN_CSV_PATH)

    x_structured_data = x_structured_data.values

    x_structured_data = x_structured_data[:num_instances]
    y_classification = test_utils.generate_one_hot_labels(
        num_instances=num_instances, num_classes=3
    )
    y_regression = test_utils.generate_data(
        num_instances=num_instances, shape=(1,)
    )

    # Build model and train.
    structured_data_input = ak.StructuredDataInput()
    structured_data_output = ak.DenseBlock()(structured_data_input)

    text_input = ak.TextInput()
    text_output = ak.TextBlock()(text_input)
    merged_outputs = ak.Merge()((structured_data_output, text_output))

    regression_outputs = ak.RegressionHead()(merged_outputs)
    classification_outputs = ak.ClassificationHead()(merged_outputs)
    automodel = ak.AutoModel(
        inputs=[text_input, structured_data_input],
        directory=tmp_path,
        outputs=[regression_outputs, classification_outputs],
        max_trials=2,
        tuner=ak.Hyperband,
        seed=test_utils.SEED,
    )

    automodel.fit(
        (x_text, x_structured_data),
        (y_regression, y_classification),
        validation_split=0.2,
        epochs=1,
        batch_size=2,
    )


def test_image_blocks(tmp_path):
    num_instances = 3
    x_train = test_utils.generate_data(
        num_instances=num_instances, shape=(28, 28)
    )
    y_train = np.random.randint(0, 10, num_instances)

    input_node = ak.ImageInput()
    output = ak.Normalization()(input_node)
    output = ak.ImageAugmentation()(output)
    outputs1 = ak.ResNetBlock(version="v2")(output)
    outputs2 = ak.XceptionBlock()(output)
    output_node = ak.Merge()((outputs1, outputs2))
    output_node = ak.ClassificationHead()(output_node)

    automodel = ak.AutoModel(
        inputs=input_node,
        outputs=output_node,
        directory=tmp_path,
        max_trials=1,
        seed=test_utils.SEED,
    )

    automodel.fit(
        x_train,
        y_train,
        validation_data=(x_train, y_train),
        epochs=1,
        batch_size=2,
    )
