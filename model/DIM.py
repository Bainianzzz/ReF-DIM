from model.NAFNet import NAFBlock
from model.conv import *


class Encoder(nn.Module):
    def __init__(self, c1=3, c_hidden=16):
        super().__init__()

        self.conv1 = Conv(c1=c1, c2=c_hidden, k=3)
        self.conv2 = Conv(c1=c_hidden, c2=c_hidden * 2, k=3, s=2)
        self.conv3 = Conv(c1=c_hidden, c2=3, k=3)

        self.pointConv1 = Conv(c1=c_hidden * 8, c2=c_hidden * 4)
        self.pointConv2 = Conv(c1=c_hidden * 16, c2=c_hidden)

        self.stage1 = C3k2(c1=c_hidden * 2, c2=c_hidden * 2, c3k=False, e=0.5)
        self.stage2 = C3k2(c1=c_hidden * 2, c2=c_hidden * 2, c3k=False, e=0.5)
        self.stage3 = C3k2(c1=c_hidden * 4, c2=c_hidden * 4, c3k=False, e=0.5)
        self.stage4 = C3k2(c1=c_hidden * 8, c2=c_hidden * 8, c3k=False, e=0.5)

        self.downsample1 = Conv(c1=c_hidden, c2=c_hidden * 2, k=2, s=2, p=0)
        self.downsample2 = Conv(c1=c_hidden * 2, c2=c_hidden * 4, k=2, s=2, p=0)
        self.downsample3 = Conv(c1=c_hidden * 4, c2=c_hidden * 8, k=2, s=2, p=0)

        self.attn1 = A2C2f(c1=c_hidden * 4, c2=c_hidden * 4, residual=True, e=1)
        self.attn2 = A2C2f(c1=c_hidden * 8, c2=c_hidden * 8, residual=True, e=0.25)

    @staticmethod
    def intensity_mapping(x):
        return x * 2 - torch.pow(x, 2)

    def forward(self, x):
        encoder_1 = self.stage1(self.conv2(self.conv1(x)))  # [B, c_hidden * 2, H/2, W/2]
        fusion_1 = self.downsample1(encoder_1)

        encoder_2 = self.stage2(fusion_1)  # [B, c_hidden * 4, H/4, W/4]
        fusion_2 = self.downsample2(encoder_2)

        encoder_3 = self.pointConv1(self.stage3(fusion_2))  # [B, c_hidden * 4, H/4, W/4]
        attn1 = self.attn1(encoder_3)
        fusion_3 = self.intensity_mapping(self.downsample3(encoder_3))

        encoder_4 = torch.cat([self.stage4(fusion_3), self.attn2(fusion_3)], 1)
        low_output = self.conv2(self.pointConv2(encoder_4))

        return low_output, encoder_4, encoder_3, encoder_2, encoder_1


class DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=16):
        super().__init__()
        self.encoder = Encoder(c1=c1, c_hidden=c_hidden)

        self.upsample1 = UpSampleConv(c1=c_hidden * 16, c2=c_hidden * 4)
        self.upsample2 = UpSampleConv(c1=c_hidden * 8, c2=c_hidden * 2)
        self.upsample3 = UpSampleConv(c1=c_hidden * 4, c2=c_hidden * 1)

        self.denoise1 = NAFBlock(c=c_hidden * 4)
        self.denoise2 = NAFBlock(c=c_hidden * 2)
        self.denoise3 = NAFBlock(c=c_hidden * 1)

        self.decoder = Conv(c1=c_hidden * 2, c2=3, k=3)

    def forward(self, x):
        low_output, encoder_4, encoder_3, encoder_2, encoder_1 = self.encoder(x)

        fusion_1 = self.denoise1(self.upsample1(encoder_4))  # [B, c_hidden*4, H/4, W/4]
        fusion_2 = self.denoise2(self.upsample2(torch.cat([fusion_1, encoder_3], 1)))  # [B, c_hidden*2, H/2, W/2]
        fusion_3 = self.denoise3(self.upsample3(torch.cat([fusion_2, encoder_2], 1)))  # [B, c_hidden, H, W]
        output = self.decoder(torch.cat([fusion_3, encoder_1], 1))  # [B, 3, H, W]

        return output, low_output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model = DIM().to(device)
    output, output_low = model(input_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params, output.size(), output_low.size())
