from keras.layers import Conv1D, Conv2D, Conv3D
from autokeras.classifier import MIN_LOSS_DEC
from autokeras.classifier import MAX_NO_IMPROVEMENT_NUM
from autokeras.classifier import MAX_ITER_NUM


def get_conv_layer_func(n_dim):
    conv_layer_functions = [Conv1D, Conv2D, Conv3D]
    if n_dim > 3:
        raise ValueError('The input dimension is too high.')
    if n_dim < 1:
        raise ValueError('The input dimension is too low.')
    return conv_layer_functions[n_dim - 1]

class ModelTrainer():
    def __init__(self, model, x_train, y_train, x_test, y_test, verbose):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.verbose = verbose
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0

    def _converged(self, loss):
        self.training_losses.append(loss)
        if loss > (self.minimum_loss - MIN_LOSS_DEC):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0

        if loss < self.minimum_loss:
            self.minimum_loss = loss

        return self._no_improvement_count > MAX_NO_IMPROVEMENT_NUM

    def train_model(self):
        self.training_losses = []
        self._no_improvement_count = 0
        self.minimum_loss = float('inf')
        for _ in range(MAX_ITER_NUM):
            self.model.fit(self.x_train, self.y_train,
                      batch_size=min(self.x_train.shape[0], 200),
                      verbose=self.verbose)
            loss, _ = self.model.evaluate(self.x_test, self.y_test)
            if self._converged(loss):
                break
        return self.minimum_loss
