from torch import nn
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class convbnrelu(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))

        return out
    
class MobileNetV3(nn.Module):
    def __init__(self):
        super().__init__()
        base = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        features = list(base.features.children())

        first_conv = features[0][0]  # Conv2d
        new_conv = nn.Conv2d(
            in_channels=first_conv.in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=1,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )

        new_conv.weight.data.copy_(first_conv.weight.data)
        if first_conv.bias is not None:
            new_conv.bias.data.copy_(first_conv.bias.data)

        features[0][0] = new_conv

        self.layer1 = nn.Sequential(*features[0:2])
        self.layer2 = nn.Sequential(*features[2:4])

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return out
