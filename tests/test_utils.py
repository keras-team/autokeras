import numpy
import subprocess
from unittest.mock import patch

from autokeras.constant import Constant
from autokeras.utils import temp_folder_generator, download_file, get_system, get_device, compute_image_resize_params, \
    resize_image_data
from tests.common import clean_dir, TEST_TEMP_DIR, mock_nvidia_smi_output


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
def test_temp_folder_generator(_):
    clean_dir(TEST_TEMP_DIR)
    path = temp_folder_generator()
    assert path == "tests/resources/temp/autokeras"
    clean_dir(TEST_TEMP_DIR)


@patch('requests.get', side_effect=mocked_requests_get)
def test_fetch(_):
    # Assert requests.get calls
    clean_dir(TEST_TEMP_DIR)
    mgc = download_file("dummy_url", TEST_TEMP_DIR + '/dummy_file')
    clean_dir(TEST_TEMP_DIR)


def test_compute_image_resize_params():
    # Case-1: Compute median height and width for smaller images.
    data = numpy.array([numpy.random.randint(256, size=(10, 10, 3)),
                        numpy.random.randint(256, size=(20, 20, 3)),
                        numpy.random.randint(256, size=(30, 30, 3)),
                        numpy.random.randint(256, size=(40, 40, 3))])
    resize_height, resize_width = compute_image_resize_params(data)
    assert resize_height == 25
    assert resize_width == 25

    modified_data = resize_image_data(data, resize_height, resize_width)
    for image in modified_data:
        assert image.shape == (25, 25, 3)

    # Case-2: Resize to max size for larger images.
    data = numpy.array([numpy.random.randint(256, size=(int(numpy.sqrt(Constant.MAX_IMAGE_SIZE)+1),
                                                        int(numpy.sqrt(Constant.MAX_IMAGE_SIZE)+1),
                                                        3))])
    resize_height, resize_width = compute_image_resize_params(data)
    assert resize_height == int(numpy.sqrt(Constant.MAX_IMAGE_SIZE))
    assert resize_width == int(numpy.sqrt(Constant.MAX_IMAGE_SIZE))

    modified_data = resize_image_data(data, resize_height, resize_width)
    for image in modified_data:
        assert image.shape == (resize_height, resize_width, 3)


def test_get_system():
    sys_name = get_system()
    assert \
        sys_name == Constant.SYS_GOOGLE_COLAB or \
        sys_name == Constant.SYS_LINUX or \
        sys_name == Constant.SYS_WINDOWS


@patch('torch.cuda.is_available')
@patch('subprocess.check_output')
def test_get_device(mock_check_output, mock_is_available):
    mock_check_output.return_value = mock_nvidia_smi_output()
    mock_is_available.return_value = True
    device = get_device()
    assert device == 'cuda:1'

    mock_check_output.side_effect = subprocess.SubprocessError
    device = get_device()
    assert device == 'cpu'

    mock_is_available.return_value = False
    mock_check_output.return_value = ''
    device = get_device()
    assert device == 'cpu'
