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

import pandas as pd

import autokeras as ak
from autokeras import test_utils


def test_io_api(tmp_path):
    num_instances = 3
    image_x = test_utils.generate_data(
        num_instances=num_instances, shape=(28, 28)
    )
    text_x = test_utils.generate_text_data(num_instances=num_instances)

    image_x = image_x[:num_instances]
    structured_data_x = (
        pd.read_csv(test_utils.TRAIN_CSV_PATH)
        .to_numpy()
        .astype(str)[:num_instances]
    )
    classification_y = test_utils.generate_one_hot_labels(
        num_instances=num_instances, num_classes=3
    )
    regression_y = test_utils.generate_data(
        num_instances=num_instances, shape=(1,)
    )

    # Build model and train.
    automodel = ak.AutoModel(
        inputs=[ak.ImageInput(), ak.TextInput(), ak.StructuredDataInput()],
        outputs=[
            ak.RegressionHead(metrics=["mae"]),
            ak.ClassificationHead(
                loss="categorical_crossentropy", metrics=["accuracy"]
            ),
        ],
        directory=tmp_path,
        max_trials=2,
        tuner=ak.RandomSearch,
        seed=test_utils.SEED,
    )
    automodel.fit(
        [image_x, text_x, structured_data_x],
        [regression_y, classification_y],
        epochs=1,
        validation_split=0.2,
        batch_size=2,
    )
    automodel.predict([image_x, text_x, structured_data_x])
