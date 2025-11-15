from model.NAFNet import NAFBlock
from model.conv import *


class Encoder(nn.Module):
    def __init__(self, c1=3, c_hidden=16):
        super().__init__()

        self.conv1 = Conv(c1=c1, c2=c_hidden, k=3)
        self.conv2 = Conv(c1=c_hidden, c2=c_hidden * 2, k=3, s=2)
        self.stage1 = C3k2(c1=c_hidden * 2, c2=c_hidden * 2, c3k=False, e=0.5)
        self.downsample1 = Conv(c1=c_hidden * 2, c2=c_hidden * 4, k=2, s=2, p=0)
        self.stage2 = C3k2(c1=c_hidden * 2, c2=c_hidden * 2, c3k=False, e=0.5)
        self.downsample2 = Conv(c1=c_hidden * 4, c2=c_hidden * 8, k=2, s=2, p=0)
        self.stage3 = C3k2(c1=c_hidden * 4, c2=c_hidden * 4, c3k=False, e=0.5)

        self.attn1 = A2C2f(c1=c_hidden * 4, c2=c_hidden * 4, e=0.5)

        self.conv3 = Conv(c1=c_hidden * 8, c2=c_hidden, k=3)
        self.conv4 = Conv(c1=c_hidden, c2=3, k=3)

    @staticmethod
    def intensity_mapping(x):
        return x * 2 - torch.pow(x, 2)

    def forward(self, x):
        encoder_1 = self.stage1(self.conv2(self.conv1(x)))  # [B, c_hidden*2, H/2, W/2]
        fusion_1 = self.downsample1(self.intensity_mapping(encoder_1))

        c_fusion1 = fusion_1.shape[1] // 2  # [B, c_hidden*4, H/4, W/4]
        encoder_2 = torch.cat([self.stage2(fusion_1[:, :c_fusion1]), fusion_1[:, c_fusion1:]], 1)
        fusion_2 = self.downsample2(self.intensity_mapping(encoder_2))

        c = fusion_2.shape[1] // 2  # [B, c_hidden*8, H/8, W/8]
        encoder_3 = torch.cat([self.stage3(fusion_2[:, :c]), self.attn1(fusion_2[:, c:])], 1)
        fusion_3 = self.intensity_mapping(encoder_3)

        low_output = self.conv4(self.conv3(fusion_3))

        return low_output, encoder_3, encoder_2, encoder_1


class DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=16):
        super().__init__()
        self.encoder = Encoder(c1=c1, c_hidden=c_hidden)

        self.upsample1 = UpSampleConv(c1=c_hidden * 8, c2=c_hidden * 2)
        self.upsample2 = UpSampleConv(c1=c_hidden * 4, c2=c_hidden * 1)
        self.upsample3 = UpSampleConv(c1=c_hidden * 2, c2=c_hidden // 2)

        self.denoise1 = NAFBlock(c=c_hidden * 2)
        self.denoise2 = NAFBlock(c=c_hidden * 1)
        self.denoise3 = NAFBlock(c=c_hidden // 2)

        self.decoder = Conv(c1=c_hidden // 2, c2=3, k=3)

    def forward(self, x):
        low_output, encoder_3, encoder_2, encoder_1 = self.encoder(x)

        fusion_1 = self.denoise1(self.upsample1(encoder_3))  # [B, c_hidden*2, H/4, W/4]

        c_ = encoder_2.shape[1] // 4 # [B, c_hidden, H/2, W/2]
        fusion_2 = self.denoise2(self.upsample2(torch.cat([fusion_1, encoder_2[:, :c_], encoder_2[:, c_*2:c_*3]], 1)))  

        c__ = encoder_1.shape[1] // 4 # [B, c_hidden, H, W]
        fusion_3 = self.denoise3(self.upsample3(torch.cat([fusion_2, encoder_1[:, :c__], encoder_1[:, c__*2:c__*3]], 1)))  # [B, 3, H, W]

        output = self.decoder(fusion_3)
        return output, low_output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(1, 3, 800, 1200).to(device)
    model = DIM().to(device)
    output, output_low = model(input_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params, output.size(), output_low.size())
