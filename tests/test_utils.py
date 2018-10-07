from unittest.mock import patch

from autokeras.utils import temp_folder_generator
from tests.common import clean_dir


@patch('tempfile.gettempdir', return_value="tests/resources/temp/")
def test_temp_folder_generator(_):
    path = 'tests/resources/temp'
    clean_dir(path)
    path = temp_folder_generator()
    assert path == "tests/resources/temp/autokeras"
    path = 'tests/resources/temp'
    clean_dir(path)

