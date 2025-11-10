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

import keras
import numpy as np
import pandas as pd

import autokeras as ak
from autokeras import test_utils

NUM_INSTANCES = 3
BATCH_SIZE = 2


def test_image_classifier(tmp_path):
    train_x = test_utils.generate_data(
        num_instances=NUM_INSTANCES, shape=(32, 32)
    )
    train_y = test_utils.generate_one_hot_labels(
        num_instances=NUM_INSTANCES, num_classes=10
    )
    clf = ak.ImageClassifier(
        directory=tmp_path,
        max_trials=2,
        seed=test_utils.SEED,
    )
    clf.fit(
        train_x, train_y, epochs=1, validation_split=0.2, batch_size=BATCH_SIZE
    )
    keras_model = clf.export_model()
    clf.evaluate(train_x, train_y)
    assert clf.predict(train_x).shape == (len(train_x), 10)
    assert isinstance(keras_model, keras.Model)


def test_image_regressor(tmp_path):
    train_x = test_utils.generate_data(
        num_instances=NUM_INSTANCES, shape=(32, 32, 3)
    )
    train_y = test_utils.generate_data(num_instances=NUM_INSTANCES, shape=(1,))
    clf = ak.ImageRegressor(
        directory=tmp_path, max_trials=2, seed=test_utils.SEED
    )
    clf.fit(
        train_x, train_y, epochs=1, validation_split=0.2, batch_size=BATCH_SIZE
    )
    clf.export_model()
    assert clf.predict(train_x).shape == (len(train_x), 1)


def test_text_classifier(tmp_path):
    train_x = test_utils.generate_text_data(num_instances=NUM_INSTANCES)
    train_y = np.array([0, 1] * ((NUM_INSTANCES + 1) // 2))[:NUM_INSTANCES]
    test_x = train_x
    test_y = train_y
    clf = ak.TextClassifier(
        directory=tmp_path,
        max_trials=2,
        seed=test_utils.SEED,
        metrics=["accuracy"],
        objective="accuracy",
    )
    clf.fit(
        train_x,
        train_y,
        epochs=1,
        validation_data=(test_x, test_y),
        batch_size=BATCH_SIZE,
    )
    clf.export_model()
    assert clf.predict(test_x).shape == (len(test_x), 1)
    assert clf.tuner._get_best_trial_epochs() <= 2


def test_text_regressor(tmp_path):
    train_x = test_utils.generate_text_data(num_instances=NUM_INSTANCES)
    test_x = train_x
    train_y = test_utils.generate_data(num_instances=NUM_INSTANCES, shape=(1,))
    test_y = train_y
    clf = ak.TextRegressor(
        directory=tmp_path, max_trials=2, seed=test_utils.SEED
    )
    clf.fit(
        train_x,
        train_y,
        epochs=1,
        validation_data=(test_x, test_y),
        batch_size=BATCH_SIZE,
    )
    clf.predict(test_x)
    clf.export_model()
    assert clf.predict(test_x).shape == (len(test_x), 1)


def test_structured_data_regressor(tmp_path):
    num_data = NUM_INSTANCES * 2
    num_train = NUM_INSTANCES
    data = (
        pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(str)[:num_data]
    )
    x_train, x_test = data[:num_train], data[num_train:]
    y = test_utils.generate_data(num_instances=num_data, shape=tuple())
    y_train, y_test = y[:num_train], y[num_train:]
    clf = ak.StructuredDataRegressor(
        directory=tmp_path, max_trials=2, seed=test_utils.SEED
    )
    clf.fit(
        x_train,
        y_train,
        epochs=11,
        validation_data=(x_train, y_train),
        batch_size=BATCH_SIZE,
    )
    clf.export_model()
    assert clf.predict(x_test).shape == (len(y_test), 1)


def test_structured_data_classifier(tmp_path):
    num_data = NUM_INSTANCES * 2
    num_train = NUM_INSTANCES
    data = (
        pd.read_csv(test_utils.TRAIN_CSV_PATH).to_numpy().astype(str)[:num_data]
    )
    x_train, x_test = data[:num_train], data[num_train:]
    y = test_utils.generate_one_hot_labels(
        num_instances=num_data, num_classes=3
    )
    y_train, y_test = y[:num_train], y[num_train:]
    clf = ak.StructuredDataClassifier(
        directory=tmp_path, max_trials=1, seed=test_utils.SEED
    )
    clf.fit(
        x_train,
        y_train,
        epochs=2,
        validation_data=(x_train, y_train),
        batch_size=BATCH_SIZE,
    )
    clf.export_model()
    assert clf.predict(x_test).shape == (len(y_test), 3)
