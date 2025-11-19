import math

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """ Pad to 'same' shape outputs. """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, bn=True, bias=True):
        super().__init__()
        self.bn = bn
        self.bias = bias

        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=self.bias)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        if self.bn is True:
            return self.act(self.bn(self.conv(x)))
        else:
            return self.forward_fuse(x)

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class UpSampleConv(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        assert c1 // 4 == c2, "The number of channels of the input and output must be 4:1"
        self.pointConv = Conv(c1, c1)
        self.upsample = nn.PixelShuffle(2)

    def forward(self, x):
        return self.upsample(self.pointConv(x))


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)  # 用于分割特征
        self.cv2 = Conv((2 + n) * self.c // 2, c2, 1)  # 最终融合层
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))  # 瓶颈层列表
        assert (2 + n) * self.c % 2 == 0

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1=(2 + n) * self.c // 2, c2=(2 + n) * self.c // 2, act=False, bn=False, bias=False),
        )

        # SimpleGate
        self.sg = SimpleGate()

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))  # 将特征分成两部分
        y.extend(m(y[-1]) for m in self.m)  # 逐步增加特征
        y_ = torch.cat(y, 1)
        y_ = self.sg(y_)
        y_ = y_ * self.sca(y_)
        return self.cv2(y_)  # 最终特征融合


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5):
        """
        Initialize the CSP Bottleneck with 3 convolutions.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c1=c_, c2=c_, act=False, bn=False, bias=False),
        )

        # SimpleGate
        self.sg = SimpleGate()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the CSP bottleneck with 3 convolutions."""
        x = torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1)
        x = self.sg(x)
        x = x * self.sca(x)
        return self.cv3(x)


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = True, g: int = 1, e: float = 0.5, k: int = 3):
        """
        Initialize C3k module.

        Args:
            c1 (int): Input channels.
            c2 (int): Output channels.
            n (int): Number of Bottleneck blocks.
            shortcut (bool): Whether to use shortcut connections.
            g (int): Groups for convolutions.
            e (float): Expansion ratio.
            k (int): Kernel size.
        """
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))