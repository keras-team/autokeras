from unittest.mock import patch

from autokeras.utils import temp_folder_generator, download_file
from tests.common import clean_dir

path = 'tests/resources/temp'


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


@patch('tempfile.gettempdir', return_value="tests/resources/temp/")
def test_temp_folder_generator(_):
    path = 'tests/resources/temp'
    clean_dir(path)
    path = temp_folder_generator()
    assert path == "tests/resources/temp/autokeras"
    path = 'tests/resources/temp'
    clean_dir(path)


@patch('requests.get', side_effect=mocked_requests_get)
def test_fetch(_):
    # Assert requests.get calls
    clean_dir(path)
    mgc = download_file("dummy_url", path + '/dummy_file')
    clean_dir(path)
