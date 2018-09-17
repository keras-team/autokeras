import os
import pickle
import torch
import tempfile
from autokeras.constant import Constant


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message


class EarlyStop:
    def __init__(self, max_no_improvement_num=Constant.MAX_NO_IMPROVEMENT_NUM, min_loss_dec=Constant.MIN_LOSS_DEC):
        super().__init__()
        self.training_losses = []
        self.minimum_loss = None
        self._no_improvement_count = 0
        self._max_no_improvement_num = max_no_improvement_num
        self._done = False
        self._min_loss_dec = min_loss_dec

    def on_train_begin(self):
        self.training_losses = []
        self._no_improvement_count = 0
        self._done = False
        self.minimum_loss = float('inf')

    def on_epoch_end(self, loss):
        self.training_losses.append(loss)
        if self._done and loss > (self.minimum_loss - self._min_loss_dec):
            return False

        if loss > (self.minimum_loss - self._min_loss_dec):
            self._no_improvement_count += 1
        else:
            self._no_improvement_count = 0
            self.minimum_loss = loss

        if self._no_improvement_count > self._max_no_improvement_num:
            self._done = True

        return True


def ensure_dir(directory):
    """Create directory if it does not exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file_dir(path):
    """Create path if it does not exist"""
    ensure_dir(os.path.dirname(path))


def has_file(path):
    return os.path.exists(path)


def pickle_from_file(path):
    return pickle.load(open(path, 'rb'))


def pickle_to_file(obj, path):
    pickle.dump(obj, open(path, 'wb'))


def get_device():
    """ If Cuda is available, use Cuda device, else use CPU device
        When choosing from Cuda devices, this function will choose the one with max memory available

    Returns: string device name

    """
    # TODO: could use gputil in the future
    if torch.cuda.is_available():
        smi_out = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU|grep Free').read()
        # smi_out=
        #       Free                 : xxxxxx MiB
        #       Free                 : xxxxxx MiB
        #                      ....
        memory_available = [int(x.split()[2]) for x in smi_out.splitlines()]
        if not memory_available:
            device = 'cpu'
        else:
            device = 'cuda:' + str(memory_available.index(max(memory_available)))
    else:
        device = 'cpu'
    return device


def temp_folder_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokeras')
    if not os.path.exists(path):
        os.makedirs(path)
    return path
