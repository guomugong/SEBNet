import torch.nn.functional as F
from model.backbone import backbone

from .seg_utils import *

class SEBNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(SEBNet, self).__init__()
        self.backbone = backbone()

        self.n_channels = n_channels
        self.n_classes = n_classes

        self.side1 = ReduceConv(64, 64)       #Stage 1, conv1, H W
        self.side2 = ReduceConvUp(128, 64, 2) #Stage 2, conv2, H/2 W/2
        self.side3 = ReduceConvUp(128, 64, 4) #Stage 3, conv3, H/4 W/4
        self.side4 = ReduceConvUp(128, 64, 8) #Stage 4, conv4, H/8 W/8
        self.side5 = ReduceConvUp(128, 64, 16)#Stage 5, conv5, H/16 W/16

        self.fusec = DoubleConv(64*5, 64)
        self.outc  = OutConv(64, n_classes)

    def forward(self, x):
        conv1, conv2, conv3, conv4, conv5 = self.backbone(x)

        side1 = self.side1(conv1)
        side2 = self.side2(conv2, x)
        side3 = self.side3(conv3, x)
        side4 = self.side4(conv4, x)
        side5 = self.side5(conv5, x)
        
        fuse = self.fusec(torch.cat([side1, side2, side3, side4, side5], dim=1))
        logits = self.outc(fuse)
        return logits
