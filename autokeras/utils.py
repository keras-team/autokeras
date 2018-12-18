import csv
import os
import pickle
import sys
import tempfile
import zipfile
import logging

import warnings
import imageio
import numpy as np
import requests
import torch
import subprocess
import string
import random
from autokeras.constant import Constant
from scipy.ndimage import zoom


class NoImprovementError(Exception):
    def __init__(self, message):
        self.message = message


def ensure_dir(directory):
    """Create directory if it does not exist."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def ensure_file_dir(path):
    """Create path if it does not exist."""
    ensure_dir(os.path.dirname(path))


def has_file(path):
    """Check if the given path exists."""
    return os.path.exists(path)


def pickle_from_file(path):
    """Load the pickle file from the provided path and returns the object."""
    return pickle.load(open(path, 'rb'))


def pickle_to_file(obj, path):
    """Save the pickle file to the specified path."""
    pickle.dump(obj, open(path, 'wb'))


# TODO cannot detect nvidia-smi in Windows normally. We need a fall back for windows
def get_device():
    """ If CUDA is available, use CUDA device, else use CPU device.
    When choosing from CUDA devices, this function will choose the one with max memory available.
    Returns: string device name.
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
            if isinstance(smi_out, bytes):
                smi_out = smi_out.decode('utf-8')
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


def temp_path_generator():
    sys_temp = tempfile.gettempdir()
    path = os.path.join(sys_temp, 'autokeras')
    return path


def rand_temp_folder_generator():
    """Create and return a temporary directory with the path name '/temp_dir_name/autokeras' (E:g:- /tmp/autokeras)."""
    chars = string.ascii_uppercase + string.digits
    size = 6
    random_suffix = ''.join(random.choice(chars) for _ in range(size))
    sys_temp = temp_path_generator()
    path = sys_temp + '_' + random_suffix
    ensure_dir(path)
    return path


def download_file(file_link, file_path):
    """Download the file specified in `file_link` and saves it in `file_path`."""
    if not os.path.exists(file_path):
        with open(file_path, "wb") as f:
            print("\nDownloading %s" % file_path)
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
    """Download the file specified in `file_link`, save to `file_path` and extract to the directory `extract_path`."""
    if not os.path.exists(extract_path):
        download_file(file_link, file_path)
        zip_ref = zipfile.ZipFile(file_path, 'r')
        print("extracting downloaded file...")
        zip_ref.extractall(extract_path)
        os.remove(file_path)
        print("extracted and removed downloaded zip file")
    print("file already extracted in the path %s" % extract_path)


def verbose_print(new_father_id, new_graph, new_model_id):
    """Print information about the operation performed on father model to obtain current model and father's id."""
    cell_size = [24, 49]

    logging.info('New Model Id - ' + str(new_model_id))
    header = ['Father Model ID', 'Added Operation']
    line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(header))
    logging.info('\n' + '+' + '-' * len(line) + '+')
    logging.info('|' + line + '|')
    logging.info('+' + '-' * len(line) + '+')
    for i in range(len(new_graph.operation_history)):
        if i == len(new_graph.operation_history) // 2:
            r = [str(new_father_id), ' '.join(str(item) for item in new_graph.operation_history[i])]
        else:
            r = [' ', ' '.join(str(item) for item in new_graph.operation_history[i])]
        line = '|'.join(str(x).center(cell_size[i]) for i, x in enumerate(r))
        logging.info('|' + line + '|')
    logging.info('+' + '-' * len(line) + '+')


def validate_xy(x_train, y_train):
    """Validate `x_train`'s type and the shape of `x_train`, `y_train`."""
    try:
        x_train = x_train.astype('float64')
    except ValueError:
        raise ValueError('x_train should only contain numerical data.')

    if len(x_train.shape) < 2:
        raise ValueError('x_train should at least has 2 dimensions.')

    if x_train.shape[0] != y_train.shape[0]:
        raise ValueError('x_train and y_train should have the same number of instances.')


def read_csv_file(csv_file_path):
    """Read the csv file and returns two separate list containing file names and their labels.

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
    """Read the image contained in the provided path `image_path`."""
    img = imageio.imread(uri=img_path)
    return img


def compute_image_resize_params(data):
    """Compute median dimension of all images in data.

    It used to resize the images later. Number of channels do not change from the original data.

    Args:
        data: 1-D, 2-D or 3-D images. The Images are expected to have channel last configuration.

    Returns:
        median shape.
    """
    if data is None or len(data.shape) == 0:
        return []

    if len(data.shape) == len(data[0].shape) + 1 and np.prod(data[0].shape[:-1]) <= Constant.MAX_IMAGE_SIZE:
        return data[0].shape

    data_shapes = []
    for x in data:
        data_shapes.append(x.shape)

    median_shape = np.median(np.array(data_shapes), axis=0)
    median_size = np.prod(median_shape[:-1])

    if median_size > Constant.MAX_IMAGE_SIZE:
        reduction_factor = np.power(Constant.MAX_IMAGE_SIZE / median_size, 1 / (len(median_shape) - 1))
        median_shape[:-1] = median_shape[:-1] * reduction_factor

    return median_shape.astype(int)


def resize_image_data(data, resize_shape):
    """Resize images to given dimension.

    Args:
        data: 1-D, 2-D or 3-D images. The Images are expected to have channel last configuration.
        resize_shape: Image resize dimension.

    Returns:
        data: Reshaped data.
    """
    if data is None or len(resize_shape) == 0:
        return data

    if len(data.shape) > 1 and np.array_equal(data[0].shape, resize_shape):
        return data

    output_data = []
    for im in data:
        output_data.append(zoom(input=im, zoom=np.divide(resize_shape, im.shape)))

    return np.array(output_data)


def get_system():
    """Get the current system environment. If the current system is not supported, raise an exception.

    Returns:
         A string to represent the current OS name.
         "posix" stands for Linux, Mac or Solaris architecture.
         "nt" stands for Windows system.
    """
    if 'google.colab' in sys.modules:
        return Constant.SYS_GOOGLE_COLAB
    if os.name == 'posix':
        return Constant.SYS_LINUX
    if os.name == 'nt':
        return Constant.SYS_WINDOWS

    raise EnvironmentError('Unsupported environment')
