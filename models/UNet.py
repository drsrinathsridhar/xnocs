import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os, sys, argparse, math
import numpy as np

FileDirPath = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(FileDirPath, '..'))
import ptUtils, ptNets

sys.path.append(os.path.join(FileDirPath, '.'))
from modules import UNet_ConvBlock, UNet_DownBlock, UNet_UpBlock

# Implemented as in the original paper with added batchnorm and relu
# Good tutorial: https://tuatini.me/practical-image-segmentation-with-unet/

class UNet(ptNets.ptNet):
    def __init__(self, in_shape, Args=None, DataParallelDevs=None):
        super().__init__(Args)

        channels, height, width = in_shape

        # Down branch
        self.DBlock1 = UNet_DownBlock(in_channels=channels, out_channels=64)
        self.DBlock2 = UNet_DownBlock(in_channels=64, out_channels=128)
        self.DBlock3 = UNet_DownBlock(in_channels=128, out_channels=256)
        self.DBlock4 = UNet_DownBlock(in_channels=256, out_channels=512)

        # Middle branch
        self.Center = nn.Sequential(
            UNet_ConvBlock(in_channels=512, out_channels=1024, kernel_size=(3, 3), stride=1, padding=0),
            UNet_ConvBlock(in_channels=1024, out_channels=1024, kernel_size=(3, 3), stride=1, padding=0)
        )

        # Up branch
        self.UBlock1 = UNet_UpBlock(in_channels=1024, out_channels=512, up_size=(56, 56))
        self.UBlock2 = UNet_UpBlock(in_channels=512, out_channels=256, up_size=(104, 104))
        self.UBlock3 = UNet_UpBlock(in_channels=256, out_channels=128, up_size=(200, 200))
        self.UBlock4 = UNet_UpBlock(in_channels=128, out_channels=64, up_size=(392, 392))

        # Final 1x1 conv with 3 output channels
        self.Output = nn.Conv2d(64, 3, kernel_size=(1, 1), stride=1, padding=0)

        if DataParallelDevs is not None:
            if len(DataParallelDevs) > 1:
                self.DBlock1 = nn.DataParallel(self.DBlock1, device_ids=DataParallelDevs)
                self.DBlock2 = nn.DataParallel(self.DBlock2, device_ids=DataParallelDevs)
                self.DBlock3 = nn.DataParallel(self.DBlock3, device_ids=DataParallelDevs)
                self.DBlock4 = nn.DataParallel(self.DBlock4, device_ids=DataParallelDevs)
                self.Center = nn.DataParallel(self.Center, device_ids=DataParallelDevs)
                self.UBlock1 = nn.DataParallel(self.UBlock1, device_ids=DataParallelDevs)
                self.UBlock2 = nn.DataParallel(self.UBlock2, device_ids=DataParallelDevs)
                self.UBlock3 = nn.DataParallel(self.UBlock3, device_ids=DataParallelDevs)
                self.UBlock4 = nn.DataParallel(self.UBlock4, device_ids=DataParallelDevs)
                self.Output = nn.DataParallel(self.Output, device_ids=DataParallelDevs)

    def forward(self, x):
        x, FM1 = self.DBlock1(x)
        x, FM2 = self.DBlock2(x)
        x, FM3 = self.DBlock3(x)
        x, FM4 = self.DBlock4(x)

        x = self.Center(x)

        x = self.UBlock1(x, FM4)
        x = self.UBlock2(x, FM3)
        x = self.UBlock3(x, FM2)
        x = self.UBlock4(x, FM1)

        Out = self.Output(x)
        Out = torch.squeeze(Out, dim=1)
        return Out