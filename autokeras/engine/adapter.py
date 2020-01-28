from autokeras.engine import picklable


class Adapter(picklable.Picklable):
    """Adpat the input and output format for Keras Model."""

    def __init__(self):
        self.shape = None

    def convert_to_dataset(self, x):
        if isinstance(x, tf.data.Dataset):
            return x
        if isinstance(x, np.ndarray):
            x = x.astype(np.float32)
            return tf.data.Dataset.from_tensor_slices(x)

    def fit(self, dataset):
        self.record_dataset_shape(dataset)

    def fit_before_convert(self, dataset):
        pass

    def fit_transform(self, dataset):
        self.check(dataset)
        self.fit_before_convert(dataset)
        dataset = self.convert_to_dataset(y)
        self.fit(dataset)
        return dataset

    def record_dataset_shape(self, dataset):
        self.shape = utils.dataset_shape(dataset)

    def transform(self, dataset):
        self.check(dataset)
        return self.convert_to_dataset(dataset)
