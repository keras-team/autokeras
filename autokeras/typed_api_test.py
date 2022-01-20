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

from typedapi import ensure_api_is_typed

import autokeras

HELP_MESSAGE = (
    "You can also take a look at this issue:\n"
    "https://github.com/keras-team/autokeras/issues/918"
)


# TODO: add types and remove all elements from
# the exception list.
EXCEPTION_LIST = [
    autokeras.BayesianOptimization,
    autokeras.CastToFloat32,
    autokeras.ExpandLastDim,
    autokeras.RandomSearch,
]


def test_api_surface_is_typed():
    ensure_api_is_typed(
        [autokeras],
        EXCEPTION_LIST,
        init_only=True,
        additional_message=HELP_MESSAGE,
    )
