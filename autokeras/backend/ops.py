# Copyright 2019 The AutoKeras Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from autokeras.backend.config import multi_backend

if multi_backend():
    from keras.src.ops import *  # noqa: F403, F401
else:
    import tensorflow as tf
    from tensorflow import cast  # noqa: F403, F401

    def any_symbolic_tensors(args=None, kwargs=None):
        args = args or ()
        kwargs = kwargs or {}
        for x in tf.nest.flatten((args, kwargs)):
            if "KerasTensor" in x.__class__.__name__:
                return True
        return False

    def shape(x):
        if any_symbolic_tensors((x,)):
            return x.shape
        dynamic = tf.shape(x)
        static = x.shape.as_list()
        return tuple(
            dynamic[i] if s is None else s for i, s in enumerate(static)
        )
