import torch
import torch.nn as nn
import torch.nn.functional as F

class HeatmapNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.conv1 = nn.Sequential(nn.Conv2d(1,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.deconv4 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.deconv3 = nn.Sequential(nn.ConvTranspose2d(256,64,4,2,1), nn.BatchNorm2d(64), nn.ReLU())
        self.deconv2 = nn.Sequential(nn.ConvTranspose2d(128,32,4,2,1), nn.BatchNorm2d(32), nn.ReLU())
        self.final = nn.Conv2d(32, num_keypoints, 1)

    def forward(self, x):
        e1 = self.conv1(x)
        e2 = self.conv2(e1)
        e3 = self.conv3(e2)
        e4 = self.conv4(e3)
        d4 = self.deconv4(e4)
        d3 = self.deconv3(torch.cat([d4, e3], dim=1))
        d2 = self.deconv2(torch.cat([d3, e2], dim=1))
        out = self.final(d2)
        return out

class RegressionNet(nn.Module):
    def __init__(self, num_keypoints=5):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.conv1 = nn.Sequential(nn.Conv2d(1,32,3,1,1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2))
        self.conv2 = nn.Sequential(nn.Conv2d(32,64,3,1,1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2))
        self.conv3 = nn.Sequential(nn.Conv2d(64,128,3,1,1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2))
        self.conv4 = nn.Sequential(nn.Conv2d(128,256,3,1,1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2))
        self.fc1 = nn.Sequential(nn.Linear(256,128), nn.ReLU(), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Dropout(0.5))
        self.fc3 = nn.Sequential(nn.Linear(64,num_keypoints*2), nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = F.adaptive_avg_pool2d(x,1).view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
