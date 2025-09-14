import torch
import torch.nn as nn
import torch.nn.functional as F

from model.conv import CSDN_Tem, C2f, Conv


class enconder_DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=18):
        super().__init__()

        self.stage1 = C2f(c1=c1, c2=c_hidden, shortcut=False)
        self.stage2 = C2f(c1=c_hidden, c2=c_hidden, shortcut=False)

    def forward(self, x):
        enconder_1 = self.stage1(x)
        enconder_2 = self.stage2(enconder_1)

        return enconder_1, enconder_2


class ReF_DIM(nn.Module):
    def __init__(self, n_task=5, c1=3, c_hidden=18):
        super().__init__()
        self.enconder = enconder_DIM(c1=c1, c_hidden=c_hidden)

        self.pointConv1 = Conv(c1=36, c2=c_hidden)
        self.pointConv2 = Conv(c1=6, c2=3)

        self.decoder = Conv(c1=c_hidden, c2=3, k=3)
        self.n_task = n_task

    def forward(self, x):
        enconder_1, enconder_2 = self.enconder(x)

        fusion = enconder_2
        IM = self.Intensity_mapping(fusion)

        fusion = self.pointConv1(torch.cat([enconder_1, fusion], 1))
        IM = self.Intensity_mapping(fusion + IM * 0.5)

        IM = self.decoder(IM)

        return self.pointConv2(torch.cat([IM, x], 1))

    def get_last_shared_layer(self):
        return self.pointConv2

    @staticmethod
    def Intensity_mapping(x):
        x = x * 2 - torch.pow(x, 2)

        return x


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(8, 3, 256, 256).to(device)
    model = ReF_DIM().to(device)
    output = model(input_tensor)

    print(output.size())
