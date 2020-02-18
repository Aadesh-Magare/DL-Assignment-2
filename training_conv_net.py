#%%
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
from torch.utils.data.sampler import SubsetRandomSampler

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms
from sklearn.metrics import confusion_matrix

random_seed = 777
torch.manual_seed(random_seed)
np.random.seed(random_seed)


class CNN(nn.Module):

    def __init__(self, n_classes=10):
        
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv4_bn = nn.BatchNorm2d(256)
        
        self.emb = nn.Linear(256*1*1, 128)
        self.emb_bn = nn.BatchNorm1d(128)
        self.emb2 = nn.Linear(128, 64)
        self.emb2_bn = nn.BatchNorm1d(64)
        self.emb3 = nn.Linear(64, 32)
        self.emb3_bn = nn.BatchNorm1d(32)
        self.clf = nn.Linear(32, n_classes)

        self.dropout = nn.Dropout(0.25)
        
    def num_flat_features(self, x):
        '''
        Calculate the total tensor x feature amount
        
        '''

        size = x.size()[1:]  # All dimensions except batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        # x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # x = self.dropout(x)
        # x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # x = self.dropout(x)
        # x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        # x = self.dropout(x)
        # x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        # x = self.dropout(x)
        
        x = F.max_pool2d(self.conv1_bn(F.relu(self.conv1(x))), 2)
        x = self.dropout(x)
        x = F.max_pool2d(self.conv2_bn(F.relu(self.conv2(x))), 2)
        x = self.dropout(x)
        x = F.max_pool2d(self.conv3_bn(F.relu(self.conv3(x))), 2)
        x = self.dropout(x)        
        x = F.max_pool2d(self.conv4_bn(F.relu(self.conv4(x))), 2)
        # x = F.max_pool2d(F.relu(self.conv4(x)), 2)
        x = self.dropout(x)        

        
        x = x.view(-1, self.num_flat_features(x))
        x = self.emb_bn(self.emb(x))
        # x = self.emb(x)
        x = self.dropout(x)
        x = self.emb2_bn(self.emb2(x))
        # x = self.emb2(x)
        x = self.dropout(x)
        x = self.emb3_bn(self.emb3(x))
        x = self.dropout(x)        
        # x = self.emb3(x)
        out = self.clf(x)

        return out

def train_one_epoch(model, trainloader, optimizer, device):
    """ 

    Training the model using the given dataloader for 1 epoch.
    Input: Model, Dataset, optimizer, 

    """

    model.train()
    losses = []
    for batch_idx, (img, target) in enumerate(trainloader):
        img = Variable(img).to(device)
        target = Variable(target).to(device)

        # Zero out the gradients
        optimizer.zero_grad()

        # Forward Propagation
        preds = model(img)
        loss = F.cross_entropy(preds, target)

        # backward propagation
        loss.backward()
        losses.append(loss.item())
        # Update the model parameters
        optimizer.step()

    return np.average(losses)

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

def evaluate():
    transform_test = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0], [0.5])
                                    ])

    dataset = FashionMNIST(os.path.join(repo_path, "./data/"), train=False, transform=transform_test, download=True)
    testloader = DataLoader(dataset, batch_size=512, shuffle=False)

    model = CNN(10)
    model.load_state_dict(torch.load(os.path.join(repo_path, "./models/CNN.pt"), map_location=torch.device('cpu')))
    loss, gt, pred = test(model, testloader)
    print("\nAccuracy on Test Data : {}\n".format(np.mean(np.array(gt) == np.array(pred))))

    cm = confusion_matrix(gt, pred)
    # print(cm)
    labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']
    import seaborn as sn
    plt.figure(figsize=(20, 10))
    sn.heatmap(cm, annot=True, cbar=True, xticklabels=labels, yticklabels=labels) # font size
    plt.savefig(os.path.join(repo_path, './img/cm_CNN.jpg'))
    # plt.show()

if __name__ == "__main__":
    repo_path = os.path.dirname(os.path.abspath(__file__))

    number_epochs = 60
    bs = 512
    valid_size = 0.15

    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    model = CNN(10).to(device)
  
    transform_train = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0], [0.5]),
                                    # transforms.Lambda(lambda x: x + 0.01 * torch.rand(x.shape))
                                    ])
  
    dataset = FashionMNIST(os.path.join(repo_path, "./data/"), train=True, transform=transform_train, download=True)
    
    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(valid_size * num_train)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    print('Train Size ', len(train_idx), 'Valid Size ', len(valid_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=train_sampler)
    validloader = torch.utils.data.DataLoader(dataset, batch_size=bs, sampler=valid_sampler)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=0, verbose=True)

    track_loss = []
    track_acc = []
    print('Training CNN Model')

    for i in tqdm(range(1, number_epochs+1)):
        model.to(device)
        loss = train_one_epoch(model, trainloader, optimizer, device)
        track_loss.append(loss)
        print('Loss: ', loss)
        if not (i % 5) :
          model.to(torch.device('cpu'))
          loss, gt, pred = test(model, validloader)
          acc = np.mean(np.array(gt) == np.array(pred))
          print("\nAccuracy on Validation Data : {}\n".format(acc))
          scheduler.step(acc)

    plt.figure()
    plt.plot(track_loss)
    plt.title("training-loss-CNN")
    plt.savefig(os.path.join(repo_path, "./img/training_loss_cnn.jpg"))

    torch.save(model.state_dict(), os.path.join(repo_path, "./models/CNN.pt"))
    print('Model saved to: ', os.path.join(repo_path, './models/CNN.pt'))

    # evaluate() 