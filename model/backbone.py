import torch
import torch.nn as nn
from model.seg_utils import ConvBNSiLUSE,MSEB


class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.conv1_1 = ConvBNSiLUSE(3, 64)
        self.conv1_2 = ConvBNSiLUSE(64, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = ConvBNSiLUSE(64, 128)
        self.conv2_2 = ConvBNSiLUSE(128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = MSEB(128, 128)
        self.conv3_2 = MSEB(128, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = MSEB(128, 128)
        self.conv4_2 = MSEB(128, 128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = MSEB(128, 128)
        self.conv5_2 = MSEB(128, 128)

    def forward(self, input):
        conv1_1 = self.conv1_1(input)
        conv1_2 = self.conv1_2(conv1_1)
        pool1 = self.pool1(conv1_2)

        conv2_1 = self.conv2_1(pool1)
        conv2_2 = self.conv2_2(conv2_1)
        pool2 = self.pool2(conv2_2)

        conv3_1 = self.conv3_1(pool2)
        conv3_2 = self.conv3_2(conv3_1)
        pool3 = self.pool3(conv3_2)

        conv4_1 = self.conv4_1(pool3)
        conv4_2 = self.conv4_2(conv4_1)
        pool4 = self.pool4(conv4_2)

        conv5_1 = self.conv5_1(pool4)
        conv5_2 = self.conv5_2(conv5_1)

        return conv1_2, conv2_2, conv3_2, conv4_2, conv5_2

def backbone():
    model = Backbone()
    return model
