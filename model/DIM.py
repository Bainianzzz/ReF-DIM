from model.NAFNet import NAFBlock
from model.conv import *


class Encoder(nn.Module):
    def __init__(self, c1=3, c_hidden=32):
        super().__init__()

        self.conv1 = Conv(c1=c1, c2=c_hidden, k=3)
        self.downsample1 = Conv(c1=c_hidden, c2=c_hidden * 2, k=2, s=2, p=0)

        self.stage1 = C3k2(c1=c_hidden * 2, c2=c_hidden * 2, c3k=False, e=0.5)
        self.downsample2 = Conv(c1=c_hidden * 2, c2=c_hidden * 4, k=2, s=2, p=0)

        self.stage2 = C3k2(c1=c_hidden * 4, c2=c_hidden * 4, c3k=False, e=0.5)
        self.downsample3 = Conv(c1=c_hidden * 4, c2=c_hidden * 8, k=2, s=2, p=0)

        self.stage3 = C3k2(c1=c_hidden * 8, c2=c_hidden * 8, c3k=False, e=0.5)
        self.attn = A2C2f(c1=c_hidden * 8, c2=c_hidden * 8, e=0.25)

        self.conv2 = Conv(c1=c_hidden * 8, c2=c_hidden, k=3)
        self.conv3 = Conv(c1=c_hidden, c2=3, k=3)

    @staticmethod
    def intensity_mapping(x):
        return x * 2 - torch.pow(x, 2)

    def forward(self, x):
        x = self.intensity_mapping(x)
        x = self.conv1(x)

        x = self.downsample1(x)
        x = self.intensity_mapping(x)
        x = self.stage1(x)
        encoder_1 = x

        x = self.downsample2(x)
        x = self.intensity_mapping(x)
        x = self.stage2(x)
        encoder_2 = x
        
        x = self.downsample3(x)
        x = self.intensity_mapping(x)
        x = self.stage3(x)
        x = self.attn(x)
        encoder_3 = x

        x = self.conv2(x)
        x = self.conv3(x)
        low_res = x

        return low_res, encoder_3, encoder_2, encoder_1


class DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=32):
        super().__init__()
        self.encoder = Encoder(c1=c1, c_hidden=c_hidden)

        self.upsample1 = UpSampleConv(c1=c_hidden * 8, c2=c_hidden * 4)
        self.upsample2 = UpSampleConv(c1=c_hidden * 4, c2=c_hidden * 2)
        self.upsample3 = UpSampleConv(c1=c_hidden * 2, c2=c_hidden)

        self.denoise0 = NAFBlock(c=c_hidden * 8)
        self.denoise1 = NAFBlock(c=c_hidden * 4)
        self.denoise2 = NAFBlock(c=c_hidden * 2)
        self.denoise3 = NAFBlock(c=c_hidden)

        self.decoder = Conv(c1=c_hidden, c2=3, k=3)

    def forward(self, x):
        low_output, encoder_3, encoder_2, encoder_1 = self.encoder(x)

        fusion_0 = self.denoise0(encoder_3)
        
        fusion_1 = self.denoise1(self.upsample1(fusion_0))  

        fusion_2 = self.denoise2(self.upsample2(fusion_1 + encoder_2))  

        fusion_3 = self.denoise3(self.upsample3(fusion_2 + encoder_1)) 

        output = self.decoder(fusion_3)
        return output, low_output


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(1, 3, 800, 1200).to(device)
    model = DIM().to(device)
    output, output_low = model(input_tensor)

    print(output.size(), output_low.size())
