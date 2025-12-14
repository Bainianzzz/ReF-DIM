import torch
import lpips
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.percep_loss = PercepLoss()

    def forward(self, input, target):
        Fidelity_Loss = self.l2_loss(input, target)
        Percep_Loss = self.percep_loss(input, target)
        return Fidelity_Loss , Percep_Loss

# Perceptual feature loss
class PercepLoss(nn.Module):
    def __init__(self):
        super(PercepLoss, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='vgg').cuda()
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, input, output):
        """
        input: shape (B, 3, H, W)
        output: shape (B, 3, H, W)
        """
        # lpips expects input, output to be in [-1, 1] and outputs (B, 1, 1, 1) -> squeeze to (B,)
        loss = self.lpips_loss(input, output)
        return loss.mean()