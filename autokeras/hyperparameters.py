from tensorflow import keras


class HyperParameter(object):

    def __init__(self, name, default=None):
        self.name = name
        self.default = default

    def get_config(self):
        return {'name': self.name, 'default': self.default}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class Choice(HyperParameter):

    def __init__(self, name, values, default=None):
        super(Choice, self).__init__(name=name, default=default)
        if not values:
            raise ValueError('`values` must be provided.')
        self.values = values

    @property
    def default_value(self):
        if self.default is not None:
            return self.default
        return self.values[0]

    def get_config(self):
        config = super(Choice, self).get_config()
        config['values'] = self.values
        return config


class Range(HyperParameter):

    def __init__(self, name, min_value, max_value, step=None, default=None):
        super(Range, self).__init__(name=name, default=default)
        self.max_value = max_value
        self.min_value = min_value
        self.step = step

    @property
    def default_value(self):
        return self.min_value

    def get_config(self):
        config = super(Range, self).get_config()
        config['max_value'] = self.max_value
        config['min_value'] = self.min_value
        config['step'] = self.step
        return config


class HyperParameters(object):

    def __init__(self):
        self.space = []
        self.values = {}

    def retrieve(self, name, type, config):
        if name in self.values:
            # TODO: type compatibility check.
            return self.values[name]
        return self.register(name, type, config)

    def register(self, name, type, config):
        config['name'] = name
        config = {'class_name': type, 'config': config}
        p = deserialize(config)
        self.space.append(p)
        value = p.default_value
        self.values[name] = value
        return value

    def get(self, name):
        if name in self.values:
            return self.values[name]
        else:
            raise ValueError('Unknown parameter: {name}'.format(name=name))

    def Choice(self, name, values, default=None):
        return self.retrieve(name, 'Choice',
                             config={'values': values,
                                     'default': default})

    def Range(self, name, min_value, max_value, step=None, default=None):
        return self.retrieve(name, 'Range',
                             config={'min_value': min_value,
                                     'max_value': max_value,
                                     'step': step,
                                     'default': default})

    def get_config(self):
        return {
            'space': [{'class_name': p.__class__.__name__,
                       'config': p.get_config()} for p in self.space],
            'values': dict((k, v) for (k, v) in self.values.items()),
        }

    @classmethod
    def from_config(cls, config):
        hp = cls()
        hp.space = [deserialize(p) for p in config['space']]
        hp.values = dict((k, v) for (k, v) in config['values'].items())
        return hp


def deserialize(config):
    module_objects = globals()
    return keras.utils.deserialize_keras_object(
        config, module_objects=module_objects)
