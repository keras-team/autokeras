from autokeras.object_detector import ObjectDetector
from tests.common import TEST_TEMP_DIR, clean_dir


def test_object_detection():
    detector = ObjectDetector()
    detector.load()
    result = detector.predict('/tests/resources/image_test/od.JPG', TEST_TEMP_DIR)
    assert isinstance(result, tuple)
    clean_dir(TEST_TEMP_DIR)
