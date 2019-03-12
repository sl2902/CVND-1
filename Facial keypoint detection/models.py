## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
import numpy as np


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # output size = (W - F)/S + 1 = (224 - 5)/1 + 1 = 220
        self.conv1 = nn.Conv2d(1, 32, 5)
        I.xavier_uniform_(self.conv1.weight, gain=np.sqrt(2)) 
        I.constant_(self.conv1.bias, 0.1)
        # after 1 pool layer, output size = (110 - 5)/1 + 1 = 106
        self.conv2 = nn.Conv2d(32, 64, 5)
        I.xavier_uniform_(self.conv2.weight, gain=np.sqrt(2))
        I.constant_(self.conv2.bias, 0.1)
        # after 2 pool layers, output size = (53 - 5)/1 + 1 = 49
        self.conv3 = nn.Conv2d(64, 128, 5)
        I.xavier_uniform_(self.conv3.weight, gain=np.sqrt(2))
        I.constant_(self.conv3.bias, 0.1)
        # after 3 pool layers, output size = (24.5 - 3)/1 + 1 = 22.5
        self.conv4 = nn.Conv2d(128, 256, 3)
        I.xavier_uniform_(self.conv4.weight, gain=np.sqrt(2))
        I.constant_(self.conv4.bias, 0.1)
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.batch_norm2 = nn.BatchNorm2d(64)
        self.batch_norm3 = nn.BatchNorm2d(128)
        self.batch_norm4 = nn.BatchNorm2d(256)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.pool = nn.MaxPool2d(2, 2)
        #self.dropout = nn.Dropout2d(p=0.75)
        # after last pool layer, output size = 11 (rounded down)
        # 256 output channels * 11.5 * 11.5 pooled map size
        self.fc1 = nn.Linear(256 * 11 * 11 , 1024)
        I.xavier_uniform_(self.fc1.weight, gain=np.sqrt(2))
        I.constant_(self.fc1.weight, 0.1)
        self.batch_norm_fc1 = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 512)
        I.xavier_uniform_(self.fc2.weight, gain=np.sqrt(2))
        I.constant_(self.fc2.weight, 0.1)
        self.batch_norm_fc2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, 136)
        I.xavier_uniform_(self.fc3.weight, gain=np.sqrt(2))
        I.constant_(self.fc3.weight, 0.1)
        
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # layer 1
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout2d(x, p=0.1)
        # layer 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout2d(x, p=0.2)
        # layer 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout2d(x, p=0.3)
        # layer 4
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = F.relu(x)
        x = self.pool(x)
        x = F.dropout2d(x, p=0.4)
        # flatten
        x = x.view(x.size(0), -1)
        # fc layer 1
        x = self.fc1(x)
        x = self.batch_norm_fc1(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=0.5)
        # fc layer 2
        x = self.fc2(x)
        x = self.batch_norm_fc2(x)
        x = F.relu(x)
        x = F.dropout2d(x, p=0.6)
        # fc layer 3
        x = self.fc3(x)
        #x = F.relu(x)
        
        
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
