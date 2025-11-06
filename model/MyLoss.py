import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Image smoothness loss (Total Variation Loss)
class L_edge(nn.Module):
    def __init__(self):
        super(L_edge, self).__init__()

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return 2 * (h_tv / count_h + w_tv / count_w) / batch_size


# Perceptual feature loss
class L_percep(nn.Module):
    def __init__(self):
        super(L_percep, self).__init__()
        self.lpips_loss = lpips.LPIPS(net='vgg').to(device)
        for param in self.lpips_loss.parameters():
            param.requires_grad = False

    def forward(self, enhanced_image, target_image):
        """
        enhanced_image: shape (B, 3, H, W)
        target_image: shape (B, 3, H, W)
        """
        # lpips expects input to be in [-1, 1] and outputs (B, 1, 1, 1) -> squeeze to (B,)
        loss = self.lpips_loss(enhanced_image, target_image)
        return loss.mean()

if __name__ == '__main__':
    model = L_percep().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    y = torch.randn(1, 3, 224, 224).cuda()
    z = model(x, y)
    print(z)
