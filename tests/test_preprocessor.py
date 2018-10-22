from unittest.mock import patch

from autokeras.cnn_module import CnnModule
from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.preprocessor import *
from tests.common import MockProcess, mock_train, clean_dir


@patch('torch.multiprocessing.Pool', new=MockProcess)
@patch('autokeras.search.ModelTrainer.train_model', side_effect=mock_train)
def test_batch_dataset(_):
    Constant.MAX_ITER_NUM = 1
    Constant.MAX_MODEL_NUM = 4
    Constant.SEARCH_MAX_ITER = 1
    Constant.T_MIN = 0.8
    data_path = 'tests/resources'
    path = 'tests/resources/temp'
    clean_dir(path)
    csv_file_path = os.path.join(data_path, "images_test/images_name.csv")
    image_path = os.path.join(data_path, "images_test/Color_images")
    train_dataset = BatchDataset(csv_file_path, image_path, has_target=True)
    test_dataset = BatchDataset(csv_file_path, image_path, has_target=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    cnn = CnnModule(classification_loss, Accuracy, {}, path, True)
    cnn.fit(2, (4, 250, 250, 3), train_dataloader, test_dataloader, 12*60*60)
    clean_dir(path)
