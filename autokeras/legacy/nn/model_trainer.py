import abc


class ModelTrainerBase(abc.ABC):
    """ A base class all model trainers will inherit from.
    Attributes:
        device: A string. Indicating the device to use. 'cuda' or 'cpu'.
        train_loader: Training data wrapped in batches in Pytorch Dataloader.
        test_loader: Testing data wrapped in batches in Pytorch Dataloader.
        loss_function: A function with two parameters (prediction, target).
            There is no specific requirement for the types of the parameters,
            as long as they are compatible with the model and the data loaders.
            The prediction should be the output of the model for a batch.
            The target should be a batch of targets packed in the data loaders.
        metric: It should be a subclass of class autokeras.metric.Metric.
            In the compute(prediction, target) function, prediction and targets are,
            all numpy arrays converted from the output of the model and the targets packed in the data loaders.
        verbose: Verbosity mode.
    """

    def __init__(self,
                 loss_function,
                 train_data,
                 test_data=None,
                 metric=None,
                 verbose=False,
                 device=None):
        self.device = device
        self.metric = metric
        self.verbose = verbose
        self.loss_function = loss_function
        self.train_loader = train_data
        self.test_loader = test_data
        self._timeout = None

    @abc.abstractmethod
    def train_model(self,
                    max_iter_num=None,
                    max_no_improvement_num=None,
                    timeout=None):
        """Train the model.
        Args:
            timeout: timeout in seconds
            max_iter_num: int, maximum numer of iteration
            max_no_improvement_num: after max_no_improvement_num,
                if the model still makes no improvement, finish training.
        """
        pass
