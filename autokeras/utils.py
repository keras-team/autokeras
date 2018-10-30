import csv
import os
import pickle
import sys
import tempfile
import zipfile

import imageio
import numpy
import requests
import torch

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
    device = 'cpu'
    if torch.cuda.is_available():
        smi_out = os.popen('nvidia-smi -q -d Memory | grep -A4 GPU|grep Free').read()
        # smi_out=
        #       Free                 : xxxxxx MiB
        #       Free                 : xxxxxx MiB
        #                      ....
        visible_devices = os.getenv('CUDA_VISIBLE_DEVICES', '').split(',')
        if len(visible_devices) == 1 and visible_devices[0] == '':
            visible_devices = []
        visible_devices = [int(x) for x in visible_devices]
        memory_available = [int(x.split()[2]) for x in smi_out.splitlines()]
        for cuda_index, _ in enumerate(memory_available):
            if cuda_index not in visible_devices and visible_devices:
                memory_available[cuda_index] = 0

        if memory_available:
            if max(memory_available) != 0:
                device = 'cuda:' + str(memory_available.index(max(memory_available)))
    return device


def temp_folder_generator():
    # return '/home/linyang/temp'
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokeras')
    ensure_dir(path)
    return path


def download_file(file_link, file_path):
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            print("Downloading %s" % file_path)
            response = requests.get(file_link, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                dl = 0
                total_length = int(total_length)
                for data in response.iter_content(chunk_size=4096):
                    dl += len(data)
                    f.write(data)
                    done = int(50 * dl / total_length)
                    sys.stdout.write("\r[%s%s]" % ('=' * done, ' ' * (50 - done)))
                    sys.stdout.flush()


def download_file_with_extract(file_link, file_path, extract_path):
    if not os.path.exists(extract_path):
        download_file(file_link, file_path)
        zip_ref = zipfile.ZipFile(file_path, 'r')
        print("extracting downloaded file...")
        zip_ref.extractall(extract_path)
        os.remove(file_path)
        print("extracted and removed downloaded zip file")
    print("file already extracted in the path %s" % extract_path)


def verbose_print(new_father_id, new_graph):
    cell_size = [24, 49]
    header = ['Father Model ID', 'Added Operation']
    line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(header))
    print('\n' + '+' + '-' * len(line) + '+')
    print('|' + line + '|')
    print('+' + '-' * len(line) + '+')
    for i in range(len(new_graph.operation_history)):
        if i == len(new_graph.operation_history) // 2:
            r = [new_father_id, new_graph.operation_history[i]]
        else:
            r = [' ', new_graph.operation_history[i]]
        line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(r))
        print('|' + line + '|')
    print('+' + '-' * len(line) + '+')


def validate_xy(x_train, y_train):
    """Check `x_train`'s type and the shape of `x_train`, `y_train`."""
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) < 2:
        raise ValueError('x_train should at least has 2 dimensions.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('x_train and y_train should have the same number of instances.')


def read_csv_file(csv_file_path):
    """Read the csv file and returns two separate list containing files name and their labels.

    Args:
        csv_file_path: Path to the CSV file.

    Returns:
        file_names: List containing files names.
        file_label: List containing their respective labels.
    """
    file_names = []
    file_labels = []
    with open(csv_file_path, 'r') as files_path:
        path_list = csv.DictReader(files_path)
        fieldnames = path_list.fieldnames
        for path in path_list:
            file_names.append(path[fieldnames[0]])
            file_labels.append(path[fieldnames[1]])
    return file_names, file_labels


def read_image(img_path):
    img = imageio.imread(uri=img_path)
    return img


def resize_image_data(data):
    """Resize each image to a fixed size H x W x C. H and W are the median height and widths computed from all images.
    Number of channels C does not change from the original image.
    """
    if len(data.shape) == 0 or len(data[0].shape) < 3:
        return data

    median_height, median_width = numpy.median(numpy.array(list(map(lambda x: x.shape, data))), axis=0)[:2]
    median_height = min(median_height.astype(int), Constant.MAX_IMAGE_HEIGHT)
    median_width = min(median_width.astype(int), Constant.MAX_IMAGE_WIDTH)
    num_of_channels = data[0].shape[-1]

    resize_data = []
    for image in data:
        resize_data.append(numpy.resize(image, (median_height, median_width, num_of_channels)))

    return numpy.array(resize_data)
