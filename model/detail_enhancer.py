import torch
import torch.nn as nn
from .block import MobileNetV3, convbnrelu
import torch.nn.functional as F

class DetailEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        mobilenetv3 = MobileNetV3()
        self.detail1 = mobilenetv3.layer1
        self.detail2 = mobilenetv3.layer2

        self.fuse1 = nn.Sequential(
            convbnrelu(24+16, 16, kernel_size=3, padding=1),
            convbnrelu(16, 16, kernel_size=3, padding=1)
        )
        self.fuse2 = nn.Sequential(
            convbnrelu(24+in_dim, 24, kernel_size=3, padding=1),
            convbnrelu(24, 24, kernel_size=3, padding=1)
        )

        self.conv_out1 = nn.Conv2d(16, 1, 1)
        self.conv_out2 = nn.Conv2d(24, 1, 1)

    def forward(self, img, scored3):
        x_d1 = self.detail1(img)
        x_d2 = self.detail2(x_d1)

        t = interpolate(scored3, x_d2.size()[2:])
        scored2 = self.fuse2(torch.cat((t, x_d2),1))

        t = interpolate(scored2, x_d1.size()[2:])
        scored1 = self.fuse1(torch.cat((t, x_d1),1))

        out1 = self.conv_out1(scored1)
        out2 = self.conv_out2(scored2)

        return out1, out2
    
interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)