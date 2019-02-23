from unittest.mock import patch

import numpy as np

from autokeras.constant import Constant
from autokeras.utils import rand_temp_folder_generator, download_file, get_system, compute_image_resize_params, \
    resize_image_data, temp_path_generator
from tests.common import clean_dir, TEST_TEMP_DIR, TEST_TEMP_AUTO_KERAS_DIR


# This method will be used by the mock to replace requests.get
def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, status_code):
            self.headers = Dic()
            self.status_code = status_code

        def iter_content(self, chunk_size=1, decode_unicode=False):
            item = "dummy".encode()
            return [item]

    class Dic:

        def __init__(self, **kwargs):
            pass

        def get(self, key):
            return 1

    if args[0] == 'dummy_url':
        return MockResponse(200)

    return MockResponse(404)


@patch('tempfile.gettempdir', return_value=TEST_TEMP_DIR)
def test_temp_path_generator(_):
    path = temp_path_generator()
    assert path == TEST_TEMP_DIR + "/autokeras"


@patch('autokeras.utils.temp_path_generator', return_value=TEST_TEMP_AUTO_KERAS_DIR)
def test_rand_temp_folder_generator(_):
    path = rand_temp_folder_generator()
    assert path.find("tests/resources/temp/autokeras_") != -1
    clean_dir(path)


@patch('requests.get', side_effect=mocked_requests_get)
def test_fetch(_):
    # Assert requests.get calls
    clean_dir(TEST_TEMP_DIR)
    mgc = download_file("dummy_url", TEST_TEMP_DIR + '/dummy_file')
    clean_dir(TEST_TEMP_DIR)


def test_compute_image_resize_params():
    # Case-1: Compute median shape for smaller (3-D) images.
    data = np.array([np.random.randint(256, size=(5, 5, 5, 3)),
                     np.random.randint(256, size=(6, 6, 6, 3)),
                     np.random.randint(256, size=(7, 7, 7, 3)),
                     np.random.randint(256, size=(8, 8, 8, 3))])
    resize_shape = compute_image_resize_params(data)
    assert np.array_equal(resize_shape, [6, 6, 6, 3])

    modified_data = resize_image_data(data, resize_shape)
    for image in modified_data:
        assert np.array_equal(image.shape, [6, 6, 6, 3])

    # Case-2: Resize to max size for larger (2-D) images.
    data = np.array([np.random.randint(256, size=(int(np.sqrt(Constant.MAX_IMAGE_SIZE) + 1),
                                                  int(np.sqrt(Constant.MAX_IMAGE_SIZE) + 1),
                                                  3))])
    resize_shape = compute_image_resize_params(data)
    assert np.array_equal(resize_shape, (int(np.sqrt(Constant.MAX_IMAGE_SIZE)),
                                         int(np.sqrt(Constant.MAX_IMAGE_SIZE)),
                                         3))

    modified_data = resize_image_data(data, resize_shape)
    for image in modified_data:
        assert np.array_equal(image.shape, resize_shape)


def test_get_system():
    sys_name = get_system()
    assert \
        sys_name == Constant.SYS_GOOGLE_COLAB or \
        sys_name == Constant.SYS_LINUX or \
        sys_name == Constant.SYS_WINDOWS


# @patch('torch.cuda.is_available')
# @patch('subprocess.check_output')
# def test_get_device(mock_check_output, mock_is_available):
#     mock_check_output.return_value = mock_nvidia_smi_output()
#     mock_is_available.return_value = True
#     device = get_device()
#     assert device == 'cuda:1'
#
#     mock_check_output.side_effect = subprocess.SubprocessError
#     device = get_device()
#     assert device == 'cpu'
#
#     mock_is_available.return_value = False
#     mock_check_output.return_value = ''
#     device = get_device()
#     assert device == 'cpu'
