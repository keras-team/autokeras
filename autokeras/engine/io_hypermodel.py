

class IOHyperModel(object):

    def get_adapter(self):
        raise NotImplementedError

    def config_from_adapter(self, adapter):
        raise NotImplementedError
