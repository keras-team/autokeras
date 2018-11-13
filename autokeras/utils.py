import csv
import os
import pickle
import sys
import tempfile
import zipfile

import warnings
import imageio
import numpy
import requests
from skimage.transform import resize
import torch
import subprocess
from autokeras.constant import Constant

from autokeras.constant import Constant


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message


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
        try:
            # smi_out=
            #       Free                 : xxxxxx MiB
            #       Free                 : xxxxxx MiB
            #                      ....
            smi_out = subprocess.check_output('nvidia-smi -q -d Memory | grep -A4 GPU|grep Free', shell=True)
            print(smi_out)
        except subprocess.SubprocessError:
            warnings.warn('Cuda device successfully detected. However, nvidia-smi cannot be invoked')
            return 'cpu'
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


def compute_image_resize_params(data):
    """Compute median height and width of all images in data. These values are used to resize the images at later point.
    Number of channels do not change from the original images. Currently, only 2-D images are supported.

    Args:
        data: 2-D Image data with shape N x H x W x C.

    Returns:
        median height: Median height of all images in the data.
        median width: Median width of all images in the data.
    """
    if len(data.shape) == 0 or len(data[0].shape) != 3:
        return None, None

    median_height, median_width = numpy.median(numpy.array(list(map(lambda x: x.shape, data))), axis=0)[:2]
    if median_height * median_width > Constant.MAX_IMAGE_SIZE:
        reduction_factor = numpy.sqrt(median_height * median_width / Constant.MAX_IMAGE_SIZE)
        median_height = median_height / reduction_factor
        median_width = median_width / reduction_factor

    return int(median_height), int(median_width)


def resize_image_data(data, h, w):
    """Resize all images in data to size (h, w, c) where h is the height, w is the width and c is the number of channels.
    The number of channels c does not change from data. The function supports only 2-D image data.

    Args:
        data: 2-D Image data with shape N x H x W x C.
        h: Image resize height.
        w: Image resize width.

    Returns:
        data: Resize data.
    """
    if data is None or h is None or w is None:
        return data

    if len(data.shape) == 0:
        return data

    output_data = []
    for im in data:
        if len(im.shape) != 3:
            return data
        output_data.append(resize(image=im,
                                  output_shape=(h, w, im.shape[-1]),
                                  mode='edge',
                                  preserve_range=True))

    return numpy.array(output_data)


def get_system():
    """
    Get the current system environment. If the current system is not supported,
    raise an exception.

    Returns:
         a string to represent the current os name
         posix stands for Linux and Mac or Solaris architecture
         nt stands for Windows system
    """
    print(os.name)
    if 'google.colab' in sys.modules:
        return Constant.SYS_GOOGLE_COLAB
    if os.name == 'posix':
        return Constant.SYS_LINUX
    if os.name == 'nt':
        return Constant.SYS_WINDOWS
    raise EnvironmentError('Unsupported environment')
