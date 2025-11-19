from model.NAFNet import NAFBlock
from model.conv import *


class Encoder(nn.Module):
    def __init__(self, c1=3, c_hidden=32):
        super().__init__()

        self.conv1 = Conv(c1=c1, c2=c_hidden, k=3)
        self.downsample1 = Conv(c1=c_hidden, c2=c_hidden * 2, k=2, s=2, p=0)  # ↓2

        self.stage1 = nn.Sequential(*(C3k(c1=c_hidden * 2, c2=c_hidden * 2, e=0.5) for _ in range(2)))
        self.downsample2 = Conv(c1=c_hidden * 2, c2=c_hidden * 4, k=2, s=2, p=0)  # ↓4

        self.stage2 = nn.Sequential(*(C3k(c1=c_hidden * 4, c2=c_hidden * 4, e=0.5) for _ in range(3)))
        self.downsample3 = Conv(c1=c_hidden * 4, c2=c_hidden * 8, k=2, s=2, p=0)  # ↓8

        self.stage3 = nn.Sequential(*(C3k(c1=c_hidden * 8, c2=c_hidden * 8, e=0.5) for _ in range(2)))

        self.low_decoder = Conv(c1=c_hidden * 8, c2=3, k=3)  # low_res ↓8

    @staticmethod
    def intensity_mapping(x):
        return x * 2 - torch.pow(x, 2)

    def forward(self, x):
        stage_1 = self.intensity_mapping(self.downsample1(self.conv1(x)))
        stage_2 = self.intensity_mapping(self.downsample2(self.stage1(stage_1)))
        stage_3 = self.intensity_mapping(self.downsample3(self.stage2(stage_2)))
        low_res = self.low_decoder(self.stage3(stage_3))

        return stage_3, stage_2, stage_1, low_res


class DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=32):
        super().__init__()
        self.encoder = Encoder(c1=c1, c_hidden=c_hidden)

        self.upsample1 = UpSampleConv(c1=c_hidden * 16, c2=c_hidden * 4)
        self.upsample2 = UpSampleConv(c1=c_hidden * 8, c2=c_hidden * 2)
        self.upsample3 = UpSampleConv(c1=c_hidden * 4, c2=c_hidden)

        self.denoise1 = nn.Sequential(*(C2f(c1=c_hidden * 8, c2=c_hidden * 8, shortcut=True, e=0.5) for _ in range(2)))
        self.denoise2 = nn.Sequential(*(C2f(c1=c_hidden * 4, c2=c_hidden * 4, shortcut=True, e=0.5) for _ in range(3)))
        self.denoise3 = nn.Sequential(*(C2f(c1=c_hidden * 2, c2=c_hidden * 2, shortcut=True, e=0.5) for _ in range(1)))

        self.decoder = Conv(c1=c_hidden, c2=3, k=3)

    def forward(self, x):
        stage_3, stage_2, stage_1, low_res = self.encoder(x)

        fusion_1 = self.upsample1(torch.cat([self.denoise1(stage_3), stage_3], 1))
        fusion_2 = self.upsample2(torch.cat([self.denoise2(fusion_1), stage_2], 1))
        fusion_3 = self.upsample3(torch.cat([self.denoise3(fusion_2), stage_1], 1))

        output = self.decoder(fusion_3)
        return output, low_res


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(1, 3, 800, 1200).to(device)
    model = DIM().to(device)
    output = model(input_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params, output.size())
