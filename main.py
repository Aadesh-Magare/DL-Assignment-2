"""
Code to use the saved models for testing
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms


def test(model, testloader):

    model.eval()

    y_gt = []
    y_pred_label = []
    losses = []

    for batch_idx, (img, y_true) in enumerate(testloader):
        img = Variable(img)
        y_true = Variable(y_true)
        out = model(img)
        y_pred = F.softmax(out, dim=1)
        y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        loss = F.cross_entropy(out, y_true)
        losses.append(loss.item())

        # Add the labels
        y_gt += list(y_true.numpy())
        y_pred_label += list(y_pred_label_tmp.numpy())

    return np.mean(losses), y_gt, y_pred_label


if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.abspath(__file__))
    trans_img = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0], [0.5])
                                    ])

    dataset = FashionMNIST(os.path.join(repo_path, "./data/"), train=False, transform=trans_img, download=True)
    testloader = DataLoader(dataset, batch_size=512, shuffle=False)

    from train_multi_layer import MLP
    model_MLP = MLP(10)
    model_MLP.load_state_dict(torch.load(os.path.join(repo_path, "./models/MLP.pt"), map_location=torch.device('cpu')))

    from training_conv_net import CNN
    model_conv_net = CNN(10)
    model_conv_net.load_state_dict(torch.load(os.path.join(repo_path, "./models/CNN.pt"), map_location=torch.device('cpu')))

    loss, gt, pred = test(model_MLP, testloader)
    with open(os.path.join(repo_path, "multi-layer-net.txt"), 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
    print('Results saved to file', 'multi-layer-net.txt')

    loss, gt, pred = test(model_conv_net, testloader)
    with open(os.path.join(repo_path, "convolution-neural-net.txt"), 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
    print('Results saved to file', 'convolution-neural-net.txt')

