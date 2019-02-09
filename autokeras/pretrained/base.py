import os

from abc import ABC, abstractmethod
from autokeras.utils import temp_path_generator, ensure_dir, download_file_from_google_drive, get_device


class Pretrained(ABC):
    """The base class for all pretrained task."""

    def __init__(self, verbose=True, model_path=None):
        """Initialize the instance."""
        self.verbose = verbose
        self.model = None
        self.device = get_device()
        self.model_path = model_path if model_path is not None else temp_path_generator()
        ensure_dir(self.model_path)
        self.local_paths = [os.path.join(self.model_path, x.local_name) for x in self._google_drive_files]
        for path, x in zip(self.local_paths, self._google_drive_files):
            if not os.path.exists(path):
                download_file_from_google_drive(file_id=x.google_drive_id,
                                                dest_path=path,
                                                verbose=True)

    @property
    @abstractmethod
    def _google_drive_files(self):
        pass

    @abstractmethod
    def predict(self, input_data, **kwargs):
        """Return predict results for the given image
        Returns:
            A numpy.ndarray containing the results.
        """
        pass
