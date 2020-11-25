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

import statistics
import sys

from benchmark import experiments as exp_module


def generate_report(experiments):
    report = [
        ",".join(
            [
                "dataset_name",
                "average_time",
                "metrics_average",
                "metrics_standard_deviation",
            ]
        )
    ]
    for experiment in experiments:
        total_times, metric_values = experiment.run(repeat_times=10)
        mean_time = statistics.mean(total_times)
        mean = statistics.mean(metric_values)
        std = statistics.stdev(metric_values)
        report.append(
            ",".join([experiment.name, str(mean_time), str(mean), str(std)])
        )
    return "\n".join(report)


def main(argv):
    task = sys.argv[1]
    path = sys.argv[2]
    report = generate_report(exp_module.get_experiments(task))
    with open(path, "w") as file:
        file.write(report)


if __name__ == "__main__":
    main(sys.argv)
