import os
import abc
import sys
from copy import deepcopy
from functools import reduce
import numpy as np
import torch
from torchvision import utils as vutils
from tqdm.autonotebook import tqdm
from autokeras.constant import Constant
from autokeras.utils import get_device


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
                 verbose=False):
        self.device = get_device()
        self.metric = metric
        self.verbose = verbose
        self.loss_function = loss_function
        self.train_loader = train_data
        self.test_loader = test_data

    @abc.abstractmethod
    def train_model(self,
                    max_iter_num=Constant.MAX_ITER_NUM,
                    max_no_improvement_num=Constant.MAX_NO_IMPROVEMENT_NUM):
        """Train the model.

        Args:
            max_iter_num: int, maximum numer of iteration
            max_no_improvement_num: after max_no_improvement_num,
                if the model still makes no improvement, finish training.
        """
        pass


class ModelTrainer(ModelTrainerBase):
    """A class that is used to train the model.

    This class can train a Pytorch model with the given data loaders.
    The metric, loss_function, and model must be compatible with each other.
    Please see the details in the Attributes.

    Attributes:
        temp_model_path: Specify the path where temp model should be stored.
        model: An instance of Pytorch Module. The model that will be trained.
        early_stop: An instance of class EarlyStop.
        optimizer: The optimizer is chosen to use the Pytorch Adam optimizer.
        current_epoch: Record the current epoch.
    """

    def __init__(self, model, path, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.model.to(self.device)
        self.optimizer = None
        self.early_stop = None
        self.current_epoch = 0
        self.current_metric_value = 0
        self.temp_model_path = os.path.join(path, 'temp_model')

    def train_model(self,
                    max_iter_num=None,
                    max_no_improvement_num=None):
        """Train the model.

        Train the model with max_iter_num or max_no_improvement_num is met.

        Args:
            max_iter_num: An integer. The maximum number of epochs to train the model.
                The training will stop when this number is reached.
            max_no_improvement_num: An integer. The maximum number of epochs when the loss value doesn't decrease.
                The training will stop when this number is reached.

        Returns:
            A tuple of loss values and metric value.
        """
        if max_iter_num is None:
            max_iter_num = Constant.MAX_ITER_NUM

        if max_no_improvement_num is None:
            max_no_improvement_num = Constant.MAX_NO_IMPROVEMENT_NUM

        self.early_stop = EarlyStop(max_no_improvement_num)
        self.early_stop.on_train_begin()

        test_metric_value_list = []
        test_loss_list = []
        self.optimizer = torch.optim.Adam(self.model.parameters())

        for epoch in range(max_iter_num):
            self._train()
            test_loss, metric_value = self._test()
            self.current_metric_value = metric_value
            test_metric_value_list.append(metric_value)
            test_loss_list.append(test_loss)
            decreasing = self.early_stop.on_epoch_end(test_loss)

            if self.early_stop.no_improvement_count == 0:
                self._save_model()

            if not decreasing:
                if self.verbose:
                    print('\nNo loss decrease after {} epochs.\n'.format(max_no_improvement_num))
                self._load_model()
                break

        last_num = min(max_no_improvement_num, max_iter_num)
        return (sum(test_loss_list[-last_num:]) / last_num,
                sum(test_metric_value_list[-last_num:]) / last_num)

    def _train(self):
        """Where the actual train proceed."""
        self.model.train()
        loader = self.train_loader
        self.current_epoch += 1

        if self.verbose:
            progress_bar = tqdm(total=len(loader),
                                desc='Epoch-'
                                     + str(self.current_epoch)
                                     + ', Current Metric - '
                                     + str(self.current_metric_value),
                                file=sys.stdout,
                                leave=False,
                                ncols=100,
                                position=0,
                                unit=' batch')
        else:
            progress_bar = None

        for batch_idx, (inputs, targets) in enumerate(deepcopy(loader)):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, targets)
            loss.backward()
            self.optimizer.step()
            if self.verbose:
                if batch_idx % 10 == 0:
                    progress_bar.update(10)
        if self.verbose:
            progress_bar.close()

    def _test(self):
        """Function for evaluation."""
        self.model.eval()
        test_loss = 0
        all_targets = []
        all_predicted = []
        loader = self.test_loader
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(deepcopy(loader)):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                # cast tensor to float
                test_loss += float(self.loss_function(outputs, targets))

                all_predicted.append(outputs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        all_predicted = reduce(lambda x, y: np.concatenate((x, y)), all_predicted)
        all_targets = reduce(lambda x, y: np.concatenate((x, y)), all_targets)
        return test_loss, self.metric.compute(all_predicted, all_targets)

    def _save_model(self):
        torch.save(self.model.state_dict(), self.temp_model_path)

    def _load_model(self):
        self.model.load_state_dict(torch.load(self.temp_model_path))


class GANModelTrainer(ModelTrainerBase):
    """A ModelTrainer especially for the GAN.

    Attributes:
        d_model: A discriminator model.
        g_model: A generator model.
        out_f: Out file.
        out_size: Size of the output image.
        optimizer_d: Optimizer for discriminator.
        optimizer_g: Optimizer for generator.
    """
    def __init__(self,
                 g_model,
                 d_model,
                 train_data,
                 loss_function,
                 verbose,
                 gen_training_result=None):
        """Initialize the GANModelTrainer.

        Args:
            g_model: The generator model to be trained.
            d_model: The discriminator model to be trained.
            train_data: the training data.
            loss_function: The loss function for both discriminator and generator.
            verbose: Whether to output the system output.
            gen_training_result: Whether to generate the intermediate result while training.
        """
        super().__init__(loss_function, train_data, verbose=verbose)
        self.d_model = d_model
        self.g_model = g_model
        self.d_model.to(self.device)
        self.g_model.to(self.device)
        self.out_f = None
        self.out_size = 0
        if gen_training_result is not None:
            self.out_f, self.out_size = gen_training_result
            self.sample_noise = torch.randn(self.out_size,
                                            self.g_model.nz,
                                            1, 1, device=self.device)
        self.optimizer_d = None
        self.optimizer_g = None

    def train_model(self,
                    max_iter_num=Constant.MAX_ITER_NUM,
                    max_no_improvement_num=Constant.MAX_NO_IMPROVEMENT_NUM):
        self.optimizer_d = torch.optim.Adam(self.d_model.parameters())
        self.optimizer_g = torch.optim.Adam(self.g_model.parameters())
        if self.verbose:
            progress_bar = tqdm(total=max_iter_num,
                                desc='     Model     ',
                                file=sys.stdout,
                                ncols=75,
                                position=1,
                                unit=' epoch')
        else:
            progress_bar = None
        for epoch in range(max_iter_num):
            self._train(epoch)
            if self.verbose:
                progress_bar.update(1)
        if self.verbose:
            progress_bar.close()

    def _train(self, epoch):
        """Perform the actual train."""
        # put model into train mode
        self.d_model.train()
        # TODO: why?
        cp_loader = deepcopy(self.train_loader)
        if self.verbose:
            progress_bar = tqdm(total=len(cp_loader),
                                desc='Current Epoch',
                                file=sys.stdout,
                                leave=False,
                                ncols=75,
                                position=0,
                                unit=' Batch')
        else:
            progress_bar = None
        real_label = 1
        fake_label = 0
        for batch_idx, inputs in enumerate(cp_loader):
            # Update Discriminator network maximize log(D(x)) + log(1 - D(G(z)))
            # train with real
            self.optimizer_d.zero_grad()
            inputs = inputs.to(self.device)
            batch_size = inputs.size(0)
            outputs = self.d_model(inputs)

            label = torch.full((batch_size,), real_label, device=self.device)
            loss_d_real = self.loss_function(outputs, label)
            loss_d_real.backward()

            # train with fake
            noise = torch.randn((batch_size, self.g_model.nz, 1, 1,), device=self.device)
            fake_outputs = self.g_model(noise)
            label.fill_(fake_label)
            outputs = self.d_model(fake_outputs.detach())
            loss_g_fake = self.loss_function(outputs, label)
            loss_g_fake.backward()
            self.optimizer_d.step()
            # (2) Update G network: maximize log(D(G(z)))
            self.g_model.zero_grad()
            label.fill_(real_label)
            outputs = self.d_model(fake_outputs)
            loss_g = self.loss_function(outputs, label)
            loss_g.backward()
            self.optimizer_g.step()

            if self.verbose:
                if batch_idx % 10 == 0:
                    progress_bar.update(10)
            if self.out_f is not None and batch_idx % 100 == 0:
                fake = self.g_model(self.sample_noise)
                vutils.save_image(
                    fake.detach(),
                    '%s/fake_samples_epoch_%03d.png' % (self.out_f, epoch),
                    normalize=True)
        if self.verbose:
            progress_bar.close()


class EarlyStop:
    """A class check for early stop condition.

    Attributes:
        training_losses: Record all the training loss.
        minimum_loss: The minimum loss we achieve so far. Used to compared to determine no improvement condition.
        no_improvement_count: Current no improvement count.
        _max_no_improvement_num: The maximum number specified.
        _done: Whether condition met.
        _min_loss_dec: A threshold for loss improvement.
    """
    def __init__(self, max_no_improvement_num=Constant.MAX_NO_IMPROVEMENT_NUM, min_loss_dec=Constant.MIN_LOSS_DEC):
        super().__init__()
        self.training_losses = []
        self.minimum_loss = None
        self.no_improvement_count = 0
        self._max_no_improvement_num = max_no_improvement_num
        self._done = False
        self._min_loss_dec = min_loss_dec

    def on_train_begin(self):
        """Initiate the early stop condition.

        Call on every time the training iteration begins.
        """
        self.training_losses = []
        self.no_improvement_count = 0
        self._done = False
        self.minimum_loss = float('inf')

    def on_epoch_end(self, loss):
        """Check the early stop condition.

        Call on every time the training iteration end.

        Args:
            loss: The loss function achieved by the epoch.

        Returns:
            True if condition met, otherwise False.
        """
        self.training_losses.append(loss)
        if self._done and loss > (self.minimum_loss - self._min_loss_dec):
            return False

        if loss > (self.minimum_loss - self._min_loss_dec):
            self.no_improvement_count += 1
        else:
            self.no_improvement_count = 0
            self.minimum_loss = loss

        if self.no_improvement_count > self._max_no_improvement_num:
            self._done = True

        return True
