import torch
import torch.nn as nn

class MultiScaleDetector(nn.Module):
    def __init__(self, num_classes=3, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.b2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.b3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
        self.b4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.head2_pre = nn.Conv2d(128, 128, 3, 1, 1)
        self.head3_pre = nn.Conv2d(256, 256, 3, 1, 1)
        self.head4_pre = nn.Conv2d(512, 512, 3, 1, 1)

        out_c = self.num_anchors * (5 + self.num_classes)
        self.head2 = nn.Conv2d(128, out_c, 1)
        self.head3 = nn.Conv2d(256, out_c, 1)
        self.head4 = nn.Conv2d(512, out_c, 1)

    def forward(self, x):
        f1 = self.b1(x)
        f2 = self.b2(f1)
        f3 = self.b3(f2)
        f4 = self.b4(f3)

        p2 = self.head2(self.head2_pre(f2))
        p3 = self.head3(self.head3_pre(f3))
        p4 = self.head4(self.head4_pre(f4))
        return [p2, p3, p4]
