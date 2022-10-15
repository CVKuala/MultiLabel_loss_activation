# Importing some important libraries

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torchvision import datasets
from torch.utils.data.sampler import SubsetRandomSampler

from loader import data_loaders
from eval_model import evaluate
import warnings
warnings.filterwarnings('ignore')

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--activation', type=str, default='sigmoid', help='final activation layer')
parser.add_argument('--loss', type=str, default='BCELoss', help='[BCELoss, CrossEntropyLoss, MultiLabelMarginLoss, MultiLabelSoftMarginLoss]')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--gamma', type=float, default=0.1, help='gamma')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
parser.add_argument('--epochs', type=int, default=10, help='no. of epochs')

args = parser.parse_args()

class BasicBlock(nn.Module):
    def __init__(self, filters, subsample=False):
        super().__init__()
        
        if(subsample == True):
            self.conv1 = nn.Conv2d(int(filters/2), filters, kernel_size=3, stride=2, padding=1, bias=False)
        else:
            self.conv1 = nn.Conv2d(int(filters), filters, kernel_size=3, stride=1, padding=1, bias=False)
            
        self.bn1   = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(filters, track_running_stats=True)
        self.relu2 = nn.ReLU()

        self.downsample = nn.AvgPool2d(kernel_size=1, stride=2)

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(layer, nn.BatchNorm2d):
                nn.init.constant_(layer.weight, 1)
                nn.init.constant_(layer.bias, 0)        
    
    def forward(self, x):
        z = self.conv1(x)
        z = self.bn1(z)
        z = self.relu1(z)
        
        z = self.conv2(z)
        z = self.bn2(z)

        z = self.relu2(z)
        
        return z
    


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.convIn = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm   = nn.BatchNorm2d(16, track_running_stats=True)
        self.relu   = nn.ReLU()
        
        self.part1 = nn.ModuleList([BasicBlock(16, subsample=False), BasicBlock(16, subsample=False), BasicBlock(16, subsample=False)])

        self.part2a = BasicBlock(32, subsample=True)
        self.part2b = nn.ModuleList([BasicBlock(32, subsample=False), BasicBlock(32, subsample=False)])

        self.part3a = BasicBlock(64, subsample=True)
        self.part3b = nn.ModuleList([BasicBlock(64, subsample=False), BasicBlock(64, subsample=False)])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.out   = nn.Linear(64, 8, bias=True)
        self.Softmax = nn.Softmax(dim=-1)
        self.Sigmoid = nn.Sigmoid()

        
        # Initilise weights
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal(layer.weight)
                layer.bias.data.zero_()      
        
        
    def forward(self, x):     
        x = self.convIn(x)
        x = self.bnorm(x)
        
        x = self.relu(x)
        for layer in self.part1: 
            x = layer(x)
        
        x = self.part2a(x)
        for layer in self.part2b: 
            x = layer(x)
        
        x = self.part3a(x)
        for layer in self.part3b: 
            x = layer(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.out(x)
        if args.activation == 'sigmoid':
            return self.Sigmoid(x)
        else:
            return self.Softmax(x)

def train_model(model, epochs, train_loader, test_loader, criterion, optimizer):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    
    train_loss_li=[]
    for epoch in range(epochs):  # loop over the dataset multiple times
        
        model.train()
        for i, data in enumerate(train_loader, 0):   # Do a batch iteration
            
            # get the inputs
            inputs, labels = data
            if(args.loss == 'MultiLabelMarginLoss' or args.loss == 'MultiLabelSoftMarginLoss'):
                labels=labels.type(torch.LongTensor)
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            # zero the parameter gradients
            optimizer.zero_grad()
            
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            # print average loss for last 3 mini-batches
            total_loss += loss.item()
            if i % 3 == 2:
                print('Epoch : {}, Batch : {}, Loss : {}'.format((epoch + 1, i + 1, total_loss / 3)))
                     
                train_loss_li.append(total_loss/3)
                total_loss = 0.0
        
        # Record metrics
        model.eval()
        train_loss = loss.item()    
    
    print('Finished Training with train_loss = {}'.format(train_loss))
    return train_loss_li

if __name__ == "__main__":
    
    # TRAINING PARAMETERS
    epochs = args.epochs
    lr = args.lr
    momentum = args.momentum
    weight_decay = args.weight_decay

    gamma = args.gamma
    loss_dict={'BCELoss':torch.nn.BCELoss(), 'CELoss': torch.nn.CrossEntropyLoss(), 'MultiLabelMarginLoss': torch.nn.MultiLabelMarginLoss(), 'MultiLabelSoftMarginLoss': torch.nn.MultiLabelSoftMarginLoss()}
    
    train_loader, test_loader = data_loaders(img_paths,data,
                                                     512,
                                                     shuffle=True,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True)

    model = ResNet()
    
    criterion = loss_dict[args.loss]
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    loss_list=train_model(model, epochs, train_loader, test_loader, criterion, optimizer)
