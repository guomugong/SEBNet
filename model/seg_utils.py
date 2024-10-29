import torch
import torch.nn as nn
import torch.nn.functional as F

class SEAttention(nn.Module):
    def __init__(self, channel=512,reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)


    def forward(self, x):
        return self.double_conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, x):
        return self.conv(x)

class ConvBNSiLU(nn.Module):
    """(convolution => [BN] => SiLU) * 1"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)

class ConvBNSiLUSE(nn.Module):
    """(convolution => [BN] => SiLU => SEAtt) * 1"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            SEAttention(out_channels),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)


class ReduceConvUp(nn.Module):
    def __init__(self, in_channels, out_channels, up_scale):
        super().__init__()
        self.convup = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Upsample(scale_factor=up_scale, mode='bilinear', align_corners=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x1, x2):
        x1 = self.convup(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        return x1


class ReduceConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.conv(x)

class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3):
        super().__init__()
        self.dsc= nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=(kernel_size-1)//2, groups=in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.dsc(x)

class SEB(nn.Module):
    def __init__(self, in_channels, out_channels, fac=6):
        super().__init__()
        self.conv0 = DSC(in_channels, out_channels)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.inc = ReduceConv(out_channels, out_channels*fac)
        self.dsc = DSC(out_channels*fac, out_channels*fac)
        self.up  = ReduceConvUp(out_channels*fac, out_channels, 2)

        self.outc = DSC(out_channels*2, out_channels)
        self.se = SEAttention(out_channels)

    def forward(self, x):
        x1 = self.conv0(x) 

        p1 = self.pool0(x1)
        p1 = self.inc(p1)
        p1 = self.dsc(p1)
        p1 = self.up(p1, x1)

        out = self.outc(torch.cat([x1,p1],dim=1))
        out = self.se(out)
        return out

class MSEB(nn.Module):
    def __init__(self, in_channels, out_channels, fac=6):
        super().__init__()
        self.conv0 = DSC(in_channels, out_channels)

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc1 = ReduceConv(out_channels, out_channels*fac)
        self.dsc1 = DSC(out_channels*fac, out_channels*fac)
        self.up1  = ReduceConvUp(out_channels*fac, out_channels, 2)

        self.fuse1 = DSC(out_channels*2, out_channels)

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc2 = ReduceConv(out_channels, out_channels*fac)
        self.dsc2 = DSC(out_channels*fac, out_channels*fac)
        self.up2  = ReduceConvUp(out_channels*fac, out_channels, 2)

        self.fuse2 = DSC(out_channels*2, out_channels)

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.inc3 = ReduceConv(out_channels, out_channels*fac)
        self.dsc3 = DSC(out_channels*fac, out_channels*fac)
        self.up3  = ReduceConvUp(out_channels*fac, out_channels, 2)

        self.outc = DSC(out_channels*2, out_channels)
        self.se = SEAttention(out_channels)

    def forward(self, x):
        x1 = self.conv0(x) 

        p1 = self.pool1(x1)
        p1 = self.inc1(p1)
        p1 = self.dsc1(p1)
        p1 = self.up1(p1, x1)

        fuse1 = self.fuse1(torch.cat([x1, p1], dim=1))

        p2 = self.pool2(fuse1)
        p2 = self.inc2(p2)
        p2 = self.dsc2(p2)
        p2 = self.up2(p2, x1)

        fuse2 = self.fuse2(torch.cat([fuse1, p2], dim=1))

        p3 = self.pool3(fuse2)
        p3 = self.inc3(p3)
        p3 = self.dsc3(p3)
        p3 = self.up3(p3, x1)

        out = self.outc(torch.cat([x1, p3], dim=1))
        out = self.se(out)
        return out
