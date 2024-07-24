import torch

def classification_accuracy(preds, targets):
    _, predicted = torch.max(preds, -1)
    # print(predicted, targets)
    correct = predicted.eq(targets).sum()
    return correct.item(), targets.size(0)

def stream_accuracy(preds, targets):
    _, predicted = torch.max(preds, 1)
    targets_pos = ~(targets == 0)
    correct = (predicted.eq(targets) * targets_pos).sum()
    return correct.item(), targets_pos.sum().item()