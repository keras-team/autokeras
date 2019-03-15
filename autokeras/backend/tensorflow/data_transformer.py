import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from autokeras.constant import Constant
from autokeras.preprocessor import DataTransformer


class ImageDataTransformer(DataTransformer):
    """ Perform basic image transformation and augmentation.

    Attributes:
        max_val: the maximum value of all data.
        mean: the mean value.
        std: the standard deviation.
        augment: whether to perform augmentation on data.
    """

    def __init__(self, data, augment=None):
        super().__init__()
        self.max_val = data.max()
        data = data / self.max_val
        self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True).flatten()
        self.std = np.std(data, axis=(0, 1, 2), keepdims=True).flatten()
        if augment is None:
            self.augment = Constant.DATA_AUGMENTATION
        else:
            self.augment = augment

    def transform_train(self, data, targets=None, batch_size=None):
        """ Transform the training data, perform random cropping data augmentation and basic random flip augmentation.

        Args:
            data: Numpy array. The data to be transformed.
            batch_size: int batch_size.
            targets: the target of training set.

        Returns:
            A DataLoader class instance.
        """

        # TODO: RandomCrop, HorizontalFlip Customize Probability, Cutout
        # channel-wise normalize the image
        data = data / self.max_val
        data = (data - self.mean)/self.std

        # other transformation
        if self.augment:
            datagen = ImageDataGenerator(
                # rescale image pixels to [0, 1]
                rescale=None,  # 1. / self.max_val,
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)
        else:
            datagen = ImageDataGenerator()

        datagen.fit(data)

        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)
        return datagen.flow(data, targets, batch_size, shuffle=True)

    def transform_test(self, data, targets=None, batch_size=None):
        """ Transform the test data, perform normalization.

        Args:
            data: Numpy array. The data to be transformed.
            batch_size: int batch_size.
            targets: the target of test set.
        Returns:
            A DataLoader instance.
        """
        # channel-wise normalize the image
        data = data / self.max_val
        data = (data - self.mean)/self.std

        datagen = ImageDataGenerator()
        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)
        return datagen.flow(data, targets, batch_size, shuffle=False)


class DataTransformerMlp(DataTransformer):
    def __init__(self, data):
        super().__init__()
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)

    def transform_train(self, data, targets=None, batch_size=None):
        data = (data - self.mean) / self.std
        data = np.nan_to_num(data)
        datagen = ImageDataGenerator()
        if batch_size is None:
            batch_size = Constant.MAX_BATCH_SIZE
        batch_size = min(len(data), batch_size)
        return datagen.flow(data, targets, batch_size, shuffle=True)

    def transform_test(self, data, target=None, batch_size=None):
        return self.transform_train(data, targets=target, batch_size=batch_size)