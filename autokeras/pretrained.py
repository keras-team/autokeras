from abc import ABC, abstractmethod
from autokeras.constant import Constant
from autokeras.image.face_detection_pretrained import detect_faces
from autokeras.utils import download_file
import os


class Pretrained(ABC):
    """The base class for all pretrained task.
    Attributes:
        verbose: A boolean value indicating the verbosity mode.
    """

    def __init__(self):
        """Initialize the instance.
        """
        self.model = None

    @abstractmethod
    def load(self):
        """load pretrained model into self.model
        """
        pass

    @abstractmethod
    def predict(self, x_predict):
        """Return predict results for the given image
        Args:
            x_predict: An instance of numpy.ndarray containing the testing data.
        Returns:
            A numpy.ndarray containing the results.
        """
        pass


class FaceDetectionPretrained(Pretrained):
    """A class to predict faces using the MTCNN pre-trained model.
    """

    def __init__(self):
        super(FaceDetectionPretrained, self).__init__()
        self.load()

    def load(self, model_path=None):
        for model_link, file_path in zip(Constant.FACE_DETECTION_PRETRAINED['PRETRAINED_MODEL_LINKS'],
                                         Constant.FACE_DETECTION_PRETRAINED['FILE_PATHS']):
            download_file(model_link, file_path)
        self.pnet, self.rnet, self.onet = Constant.FACE_DETECTION_PRETRAINED['FILE_PATHS']

    def predict(self, img_path, output_file_path=None):
        """Predicts faces in an image.

        Args:
            img_path: A string. The path to the image on which the prediction is to be done.
            output_file_path: A string. The path where the output image is to be saved after the prediction. `None` by default.

        Returns:
            A tuple containing numpy arrays of bounding boxes and landmarks. Bounding boxes are of shape `(n, 5)` and
            landmarks are of shape `(n, 10)` where `n` is the number of faces predicted. Each bounding box is of length
            5 and the corresponding rectangle is defined by the first four values. Each bounding box has five landmarks
            represented by 10 coordinates.
        """
        if not os.path.exists(img_path):
            raise ValueError('Image does not exist')
        return detect_faces(self.pnet, self.rnet, self.onet, img_path, output_file_path)
