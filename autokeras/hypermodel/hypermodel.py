

class HyperModel(object):

    def __init__(self, name=None, tunable=True):
        self.name = name
        self.tunable = tunable

    def __call__(self, *args, **kwargs):
        pass

    def build(self, hp):
        raise NotImplementedError


class DefaultHyperModel(HyperModel):

    def __init__(self, build, name=None, tunable=True):
        super(DefaultHyperModel, self).__init__(name=name, tunable=tunable)
        self.build = build
