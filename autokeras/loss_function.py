import torch


def classification_loss(prediction, target):
    labels = target.argmax(1)
    return torch.nn.NLLLoss()(prediction, labels)
