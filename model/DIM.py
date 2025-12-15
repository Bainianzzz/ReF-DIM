import torch
import torch.nn as nn
from model.conv import *
from model.loss import LossFunction

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class EncoderBlock(nn.Module):
    def __init__(self, c1=3, c_hidden=16):
        super().__init__()
        self.conv_in = Conv(c1=c1, c2=c_hidden, k=3)
        self.conv_out = Conv(c1=c_hidden // 2, c2=3, k=3)

        self.stage = C3k2(c1=c_hidden, c2=c_hidden, c3k=False, e=0.5)
        self.sg = SimpleGate()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c_hidden // 2, out_channels=c_hidden // 2, kernel_size=1),
        )

    @staticmethod
    def intensity_mapping(x):
        return x * 2 - torch.pow(x, 2)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.stage(x)
        x = self.intensity_mapping(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv_out(x)
        return x


class DIM(nn.Module):
    def __init__(self, c1=3, c_hidden=16, range=6, weights=[1, 0.2]):
        super().__init__()
        self.range = range
        self.weights = weights
        self.enhance = EncoderBlock(c1=c1, c_hidden=c_hidden)
        self.loss_fn = LossFunction()

    def forward(self, x):
        outputs = []

        for i in range(self.range):
            x = self.enhance(x)
            outputs.append(x)

        return outputs

    def _loss(self, input, target):
        outputs = self.forward(input)
        total_loss = 0
        L2_Loss = 0
        Percep_Loss = 0
        for i in range(self.range):
            stage_target = input + (target - input)*i / self.range
            loss = self.loss_fn(outputs[i], stage_target)
            total_loss += loss[0] * self.weights[0] + loss[1] * self.weights[1]
            L2_Loss += loss[0] * self.weights[0]
            Percep_Loss += loss[1] * self.weights[1]
        return total_loss, L2_Loss, Percep_Loss


class DIMInference(nn.Module):
    """
    DIM model version for validation/inference, without the loss function.
    Can load weights from a DIM training model checkpoint (will automatically skip loss_fn keys).
    """
    def __init__(self, c1=3, c_hidden=16, range=6):
        super().__init__()
        self.range = range
        self.enhance = EncoderBlock(c1=c1, c_hidden=c_hidden)

    def forward(self, x):
        outputs = []
        for i in range(self.range):
            x = self.enhance(x)
            outputs.append(x)
        return outputs
    
    def load_from_dim(self, dim_state_dict, strict=False):
        inference_state_dict = {k: v for k, v in dim_state_dict.items() 
                               if not k.startswith('loss_fn.')}
        return self.load_state_dict(inference_state_dict, strict=strict)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    input_tensor = torch.randn(1, 3, 256, 256).to(device)
    model = DIM().to(device)
    output, output_low = model(input_tensor)

    total_params = sum(p.numel() for p in model.parameters())
    print(total_params, output.size(), output_low.size())
