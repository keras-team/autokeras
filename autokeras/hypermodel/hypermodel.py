import tensorflow as tf
import types


class HyperModel(object):

    def __init__(self, name=None, tunable=True):
        if not name:
            prefix = self.__class__.__name__
            name = prefix + '_' + str(tf.keras.backend.get_uid(prefix))
        self.name = name
        self.tunable = tunable

    def __new__(cls, *args, **kwargs):
        obj = super().__new__(cls)
        build_fn = obj.build

        def build_wrapper(obj, *args, **kwargs):
            with tf.name_scope(obj.name):
                return build_fn(*args, **kwargs)

        obj.build = types.MethodType(build_wrapper, obj)
        return obj

    def build(self, hp, **kwargs):
        raise NotImplementedError


class DefaultHyperModel(HyperModel):

    def build(self, hp, **kwargs):
        pass

    def __init__(self, build, name=None, tunable=True):
        super(DefaultHyperModel, self).__init__(name=name, tunable=tunable)
        self.build = build
