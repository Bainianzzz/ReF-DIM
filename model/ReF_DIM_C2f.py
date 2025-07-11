import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv import CSDN_Tem, C2f, Conv


class enconder_DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=18):
        super().__init__()

        self.stage1 = C2f(c1=c1, c2=c_hidden, shortcut=False)
        self.stage2 = C2f(c1=c_hidden, c2=c_hidden, shortcut=False)
        # self.stage3 = C2f(c1=c_hidden, c2=c_hidden, shortcut=True)

    def forward(self, x):
        # x = self.Intensity_mapping(x)
        enconder_1 = self.stage1(x)

        # enconder_1 = self.Intensity_mapping(enconder_1)
        enconder_2 = self.stage2(enconder_1)
        # enconder_3 = self.stage3(enconder_2)

        # return enconder_1, enconder_2, enconder_3
        return enconder_1, enconder_2

    @staticmethod
    def Intensity_mapping(x):
        x = x * 2 - torch.pow(x, 2)

        return x


class ReF_DIM(nn.Module):
    def __init__(self, n_task=5, c1=3, c_hidden=18):
        super().__init__()
        self.enconder = enconder_DIM(c1=c1, c_hidden=c_hidden)

        # self.dwConv1 = CSDN_Tem(in_ch=36, out_ch=18)
        # self.dwConv2 = CSDN_Tem(in_ch=36, out_ch=18)

        self.pointConv1 = Conv(c1=36, c2=18)
        self.pointConv2 = Conv(c1=6, c2=3)

        self.decoder = Conv(c1=c_hidden, c2=3, k=3)
        self.n_task = n_task

    def forward(self, x):
        # enconder_1, enconder_2, enconder_3 = self.enconder(x)
        enconder_1, enconder_2 = self.enconder(x)

        # fusion = enconder_3
        # IM = self.Intensity_mapping(fusion)
        #
        # fusion = self.pointConv1(torch.cat([enconder_2, fusion], 1))
        # IM = self.Intensity_mapping(fusion + IM)
        #
        # fusion = self.pointConv2(torch.cat([enconder_1, fusion], 1))
        # IM = self.Intensity_mapping(fusion + IM)

        fusion = enconder_2
        IM = self.Intensity_mapping(fusion)

        fusion = self.pointConv1(torch.cat([enconder_1, fusion], 1))
        IM = self.Intensity_mapping(fusion + IM)

        IM = self.decoder(IM)

        return self.pointConv2(torch.cat([IM, x], 1))

    @staticmethod
    def Intensity_mapping(x):
        x = x * 2 - torch.pow(x, 2)

        return x


if __name__ == "__main__":
    device = 'cuda'

    input_tensor = torch.randn(8, 3, 256, 256).to(device)
    model = ReF_DIM().to(device)
    output = model(input_tensor)

    print("输出张量的形状:", output.size())
