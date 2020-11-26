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

import shutil
import timeit


class Experiment(object):
    def __init__(self, name, tmp_dir="tmp_dir"):
        self.name = name
        self.tmp_dir = tmp_dir

    def get_auto_model(self):
        raise NotImplementedError

    @staticmethod
    def load_data():
        raise NotImplementedError

    def run_once(self):
        (x_train, y_train), (x_test, y_test) = self.load_data()
        auto_model = self.get_auto_model()

        start_time = timeit.default_timer()
        auto_model.fit(x_train, y_train)
        stop_time = timeit.default_timer()

        accuracy = auto_model.evaluate(x_test, y_test)[1]
        total_time = stop_time - start_time

        return total_time, accuracy

    def run(self, repeat_times=1):
        total_times = []
        metric_values = []
        for i in range(repeat_times):
            total_time, metric = self.run_once()
            total_times.append(total_time)
            metric_values.append(metric)
            self.tear_down()
        return total_times, metric_values

    def tear_down(self):
        shutil.rmtree(self.tmp_dir)
