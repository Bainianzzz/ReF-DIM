import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import Conv


class enconder_DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=18):
        super().__init__()
        self.stage1 = Conv(c1=c1, c2=c_hidden, k=3, bn=True, bias=True)
        self.stage2 = Conv(c1=c_hidden, c2=c_hidden, k=3, bn=True, bias=True)
        self.stage3 = Conv(c1=c_hidden, c2=c_hidden, k=3, bn=True, bias=True)
        self.stage4 = Conv(c1=c_hidden, c2=c_hidden, k=3, bn=True, bias=True)
        self.stage5 = Conv(c1=c_hidden, c2=c_hidden, k=3, bn=True, bias=True)
        self.stage6 = Conv(c1=c_hidden, c2=c_hidden, k=3, bn=True, bias=True)

    def forward(self, x):
        enconder_1 = self.stage1(x)
        enconder_2 = self.stage2(enconder_1)
        enconder_3 = self.stage3(enconder_2)
        enconder_4 = self.stage4(enconder_3)
        enconder_5 = self.stage5(enconder_4)
        enconder_6 = self.stage6(enconder_5)

        return enconder_1, enconder_2, enconder_3, enconder_4, enconder_5, enconder_6


class ReF_DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=18):
        super().__init__()
        self.enconder = enconder_DIM(c1=c1, c_hidden=c_hidden)
        self.pointConv1 = Conv(c1=c_hidden * 2, c2=c_hidden)
        self.pointConv2 = Conv(c1=c_hidden * 2, c2=c_hidden)
        self.pointConv3 = Conv(c1=c_hidden * 2, c2=c_hidden)
        self.pointConv4 = Conv(c1=c_hidden * 2, c2=c_hidden)
        self.pointConv5 = Conv(c1=c_hidden * 2, c2=c_hidden)

        self.decoder = Conv(c1=c_hidden, c2=3, k=3)

    def forward(self, x):
        enconder_1, enconder_2, enconder_3, enconder_4, enconder_5, enconder_6 = self.enconder(x)

        IM = self.Intensity_mapping(enconder_6)

        fusion = self.pointConv1(torch.cat([enconder_5, IM], 1))
        IM = self.Intensity_mapping(IM)

        fusion = self.pointConv2(torch.cat([enconder_4, fusion + IM], 1))
        IM = self.Intensity_mapping(IM)

        fusion = self.pointConv3(torch.cat([enconder_3, fusion + IM], 1))
        IM = self.Intensity_mapping(IM)

        fusion = self.pointConv4(torch.cat([enconder_2, fusion + IM], 1))
        IM = self.Intensity_mapping(IM)

        fusion = self.pointConv5(torch.cat([enconder_1, fusion + IM], 1))

        IM = self.decoder(self.Intensity_mapping(IM + fusion))

        return IM

    @staticmethod
    def Intensity_mapping(x):
        x = x * 2 - torch.pow(x, 2)

        return x


if __name__ == "__main__":
    device = 'cuda'

    input_tensor = torch.randn(1, 3, 512, 512).to(device)
    model = ReF_DIM().to(device)
    output = model(input_tensor)

    print("输出张量的形状:", output.shape)
