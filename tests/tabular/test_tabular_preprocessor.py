from unittest.mock import patch

import pytest

from autokeras.tabular.tabular_preprocessor import *

from tests.common import clean_dir, TEST_TEMP_DIR