import torch
import torch.nn as nn
import torch.nn.functional as F

from .pvt_v2 import *
from .mask_decoder import MaskDecoder
from .detail_enhancer import DetailEnhancer


class SurfSyn(nn.Module):
    def __init__(self, version) -> None:
        super(SurfSyn, self).__init__()
        # encoder
        model_versions = {
            'SurfSyn-T': pvt_v2_b2(),
            'SurfSyn-S': pvt_v2_b3(),
            'SurfSyn-B': pvt_v2_b4(),
            'SurfSyn-L': pvt_v2_b5()
        }
        self.encoder = model_versions[version]

        # mask decoder
        self.mask_decoder = MaskDecoder([64, 128, 320, 512])

        # detail enhancer
        self.detail_enhancer = DetailEnhancer(64)

    def forward(self, x):
        # image encoder
        x1, x2, x3, x4 = self.encoder(x)

        # mask decoder
        out3, out4, out5, scored3 = self.mask_decoder(x1, x2, x3, x4)

        # detail enhancer
        out1, out2 = self.detail_enhancer(x, scored3)

        # upsampling
        out1 = interpolate(out1, x.size()[2:])
        out2 = interpolate(out2, x.size()[2:])
        out3 = interpolate(out3, x.size()[2:])
        out4 = interpolate(out4, x.size()[2:])
        out5 = interpolate(out5, x.size()[2:])

        return torch.sigmoid(out1), torch.sigmoid(out2), torch.sigmoid(out3), torch.sigmoid(out4), torch.sigmoid(out5)
    
interpolate = lambda x, size: F.interpolate(x, size=size, mode='bilinear', align_corners=True)