from autokeras.pretrained import FaceDetectionPretrained
import os
import urllib.request


def test_face_detection():
    img_file, out_file = 'test.jpg', 'output.jpg'
    urllib.request.urlretrieve('https://raw.githubusercontent.com/kuaikuaikim/DFace/master/test.jpg', img_file)
    if os.path.exists(out_file):
        os.remove(out_file)
    face_detection = FaceDetectionPretrained()
    bboxs1, landmarks1 = face_detection.predict(img_file, out_file)
    assert os.path.exists(out_file)
    bboxs2, landmarks2 = face_detection.predict(img_file)
    assert bboxs1.shape == bboxs2.shape == (11, 5) and landmarks1.shape == landmarks2.shape == (11, 10)
    os.remove(img_file)
    os.remove(out_file)
