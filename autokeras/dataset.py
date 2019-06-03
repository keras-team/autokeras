from sklearn.model_selection import train_test_split

from autokeras.constant import Constant


class Dataset(object):
    def __init__(self,
                 x_train=None,
                 y_train=None,
                 x_valid=None,
                 y_valid=None,
                 x_test=None,
                 y_test=None):
        self.x_train = x_train
        self.y_train = y_train
        self.x_valid = x_valid
        self.y_valid = y_valid
        self.x_test = x_test
        self.y_test = y_test

    def split_train_to_valid(self):
        # Generate split index
        dataset = [data for data in [self.x_train[0], self.y_train[0]] if data is not None]
        train_index = None
        valid_index = None
        for data in dataset:
            validation_set_size = int(len(data) * Constant.VALIDATION_SET_SIZE)
            validation_set_size = min(validation_set_size, 500)
            validation_set_size = max(validation_set_size, 1)
            train_index, valid_index = train_test_split(range(len(data)),
                                                        test_size=validation_set_size,
                                                        random_state=Constant.SEED)
            break

        # Split the data
        if self.x_train is not None:
            temp_x_train = self.x_train
            for temp_x_train_input in temp_x_train:
                self.x_train, self.x_valid = temp_x_train_input[train_index], temp_x_train_input[valid_index]
        if self.y_train is not None:
            temp_y_train = self.y_train
            for temp_y_train_input in temp_y_train:
                self.y_train, self.y_valid = temp_y_train_input[train_index], temp_y_train_input[valid_index]

