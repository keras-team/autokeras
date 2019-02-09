from autokeras.pretrained.object_detector import ObjectDetector
from tests.common import TEST_TEMP_DIR, clean_dir


def test_object_detection():
    detector = ObjectDetector()
    img_path = 'tests/resources/images_test/od.JPG'
    result = detector.predict(img_path, TEST_TEMP_DIR)
    assert isinstance(result, list)
    clean_dir(TEST_TEMP_DIR)
