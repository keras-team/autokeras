"""
Run NAS baseline methods
========================
We provide 4 NAS baseline methods now, the default one is bayesian optimization.
Here is a tutorial about running NAS baseline methods.

Generally, to run a non-default NAS methods, we will do the following steps in order:
1. Prepare the dataset in the form of torch.utils.data.DataLoader.
2. Initialize the CnnModule/MlpModule with the class name of the NAS Searcher.
3. Start search by running fit function.
Refer the cifar10 example below for more details.
"""
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.nn.functional import cross_entropy

from autokeras import CnnModule
from autokeras.nn.metric import Accuracy
from nas.greedy import GreedySearcher

if __name__ == '__main__':
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                             shuffle=False, num_workers=2)
    (image, target) = trainset[0]
    image = np.array(image).transpose((1, 2, 0))
    # add dim for batch
    input_shape = np.expand_dims(image, axis=0).shape
    num_classes = 10

    # take GreedySearcher as an example, you can implement your own searcher and
    # pass the class name to the CnnModule by search_type=YOUR_SEARCHER.
    cnnModule = CnnModule(loss=cross_entropy, metric=Accuracy,
                          searcher_args={}, verbose=True,
                          search_type=GreedySearcher)

    cnnModule.fit(n_output_node=num_classes, input_shape=input_shape,
                  train_data=trainloader,
                  test_data=testloader)