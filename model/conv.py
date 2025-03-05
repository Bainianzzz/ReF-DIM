import numpy as np
import torch
import torch.nn as nn


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
