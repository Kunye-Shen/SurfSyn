import torch
import torch.nn as nn
from .block import convbnrelu
import torch.nn.functional as F


class MaskDecoder(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # fuse
        self.fuse3 = nn.Sequential(
            convbnrelu(channels[0]+channels[0], channels[0], kernel_size=3, padding=1),
            convbnrelu(channels[0], channels[0], kernel_size=3, padding=1)
        )
        self.fuse4 = nn.Sequential(
            convbnrelu(channels[1]+channels[1], channels[0], kernel_size=3, padding=1),
            convbnrelu(channels[0], channels[0], kernel_size=3, padding=1)
        )
        self.fuse5 = nn.Sequential(
            convbnrelu(channels[3]+channels[2], channels[1], kernel_size=3, padding=1),
            convbnrelu(channels[1], channels[1], kernel_size=3, padding=1)
        )

        # output
        self.conv_out3 = nn.Conv2d(channels[0], 1, 1)
        self.conv_out4 = nn.Conv2d(channels[0], 1, 1)
        self.conv_out5 = nn.Conv2d(channels[1], 1, 1)

    def forward(self, x1, x2, x3, x4):
        # fuse
        t = interpolate(x4, x3.size()[2:])
        scored5 = self.fuse5(torch.cat((t, x3),1))

        t = interpolate(scored5, x2.size()[2:])
        scored4 = self.fuse4(torch.cat((t, x2),1))

        t = interpolate(scored4, x1.size()[2:])
        scored3 = self.fuse3(torch.cat((t, x1),1))

        # output
        out3 = self.conv_out3(scored3)
        out4 = self.conv_out4(scored4)
        out5 = self.conv_out5(scored5)

        return out3, out4, out5, scored3
    
interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)