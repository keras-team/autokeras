import subprocess
from unittest.mock import patch

from autokeras.constant import Constant
from autokeras.utils import temp_folder_generator, download_file, get_system, get_device
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
