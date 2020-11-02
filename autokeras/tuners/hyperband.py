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

import kerastuner

from autokeras.engine import tuner as tuner_module


class Hyperband(kerastuner.Hyperband, tuner_module.AutoTuner):
    """KerasTuner Hyperband with preprocessing layer tuning."""

    def __init__(
        self, max_epochs: int = 1000, max_trials: int = 100, *args, **kwargs
    ):
        super().__init__(max_epochs=max_epochs, *args, **kwargs)
        self.oracle.max_trials = max_trials
