from autokeras.pretrained import FaceDetectionPretrained
import os
import urllib.request


def test_face_detection():
    img_file, out_file = 'test.jpg', 'output.jpg'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/kuaikuaikim/DFace/master/test.jpg', img_file)
    if os.path.exists(out_file):
        os.remove(out_file)
    face_detection = FaceDetectionPretrained()
    bboxs, landmarks = face_detection.predict(img_file, out_file)
    assert os.path.exists(out_file)
    os.remove(img_file)
    os.remove(out_file)
    assert bboxs.shape == (11, 5) and landmarks.shape == (11, 10)
