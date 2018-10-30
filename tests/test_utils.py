from unittest.mock import patch

from autokeras.utils import temp_folder_generator, download_file, resize_image_data
from tests.common import clean_dir

import numpy

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


def test_resize_image_data():
    data = numpy.array([numpy.random.randint(256, size=(1, 1, 3)),
                        numpy.random.randint(256, size=(2, 2, 3)),
                        numpy.random.randint(256, size=(3, 3, 3)),
                        numpy.random.randint(256, size=(4, 4, 3))])

    data = resize_image_data(data)

    assert data[0].shape == (2, 2, 3)
    for i in range(len(data)-1):
        assert data[i].shape == data[i+1].shape
