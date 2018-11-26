from functools import reduce

import torch

import numpy as np
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from autokeras.nn.loss_function import classification_loss
from autokeras.nn.metric import Accuracy
from autokeras.nn.model_trainer import ModelTrainer
from autokeras.preprocessor import OneHotEncoder, MultiTransformDataset


class Net(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


model = Net(50, 100, 10)
n_instance = 100
batch_size = 32

train_x = np.random.random((n_instance, 50))
test_x = np.random.random((n_instance, 50))
train_y = np.random.randint(0, 9, n_instance)
test_y = np.random.randint(0, 9, n_instance)
print(train_x.shape)
print(train_y.shape)

encoder = OneHotEncoder()
encoder.fit(train_y)
train_y = encoder.transform(train_y)
test_y = encoder.transform(test_y)

compose_list = Compose([])
train_data = DataLoader(MultiTransformDataset(torch.Tensor(train_x), torch.Tensor(train_y), compose_list), batch_size=batch_size, shuffle=False)
test_data = DataLoader(MultiTransformDataset(torch.Tensor(test_x), torch.Tensor(test_y), compose_list), batch_size=batch_size, shuffle=False)

model_trainer = ModelTrainer(model,
                             loss_function=classification_loss,
                             metric=Accuracy,
                             train_data=train_data,
                             test_data=test_data,
                             verbose=True)

model_trainer.train_model(2, 1)
model.eval()

outputs = []
with torch.no_grad():
    for index, (inputs, _) in enumerate(test_data):
        outputs.append(model(inputs).numpy())
output = reduce(lambda x, y: np.concatenate((x, y)), outputs)
predicted = encoder.inverse_transform(output)
print(predicted)

