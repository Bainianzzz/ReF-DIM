import torch
import lpips
import torch.nn as nn

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

class SmoothLoss(nn.Module):
    def __init__(self):
        super(SmoothLoss, self).__init__()
        self.sigma = 10

    def rgb2yCbCr(self, input_im):
        """
        Convert an RGB image to YCbCr color space.
        """
        im_flat = input_im.contiguous().view(-1, 3).float()
        mat = torch.Tensor([[0.257, -0.148, 0.439], [0.564, -0.291, -0.368], [0.098, 0.439, -0.071]]).cuda()
        bias = torch.Tensor([16.0 / 255.0, 128.0 / 255.0, 128.0 / 255.0]).cuda()
        temp = im_flat.mm(mat) + bias
        out = temp.view(input_im.shape[0], 3, input_im.shape[2], input_im.shape[3])
        return out

    def forward(self, input, output):
        input_ycbcr = self.rgb2yCbCr(input)
        sigma_color = -1.0 / (2 * self.sigma * self.sigma)
        p = 1.0
        
        # Define 24 offset directions for slicing: 
        # [(h1_start, h1_end, w1_start, w1_end, h2_start, h2_end, w2_start, w2_end), ...]
        # Negative numbers indicate backwards offsets from the end
        directions = [
            # 8 directions with a distance of 1
            (1, None, 0, None, -1, None, 0, None),
            (-1, None, 0, None, 1, None, 0, None),
            (0, None, 1, None, 0, None, -1, None),
            (0, None, -1, None, 0, None, 1, None),
            (1, None, 1, None, -1, None, -1, None),
            (-1, None, -1, None, 1, None, 1, None),
            (1, None, -1, None, -1, None, 1, None),
            (-1, None, 1, None, 1, None, -1, None),
            # 16 directions with a distance of 2
            (2, None, 0, None, -2, None, 0, None),
            (-2, None, 0, None, 2, None, 0, None),
            (0, None, 2, None, 0, None, -2, None),
            (0, None, -2, None, 0, None, 2, None),
            (2, None, 1, None, -2, None, -1, None),
            (-2, None, -1, None, 2, None, 1, None),
            (2, None, -1, None, -2, None, 1, None),
            (-2, None, 1, None, 2, None, -1, None),
            (1, None, 2, None, -1, None, -2, None),
            (-1, None, -2, None, 1, None, 2, None),
            (1, None, -2, None, -1, None, 2, None),
            (-1, None, 2, None, 1, None, -2, None),
            (2, None, 2, None, -2, None, -2, None),
            (-2, None, -2, None, 2, None, 2, None),
            (2, None, -2, None, -2, None, 2, None),
            (-2, None, 2, None, 2, None, -2, None),
        ]
        
        total_loss = 0.0
        for h1_s, h1_e, w1_s, w1_e, h2_s, h2_e, w2_s, w2_e in directions:
            # Construct slices for each direction
            h1 = slice(h1_s, h1_e) if h1_s != 0 else slice(h1_e, h1_s) if h1_e != 0 else slice(None)
            h2 = slice(h2_s, h2_e) if h2_s != 0 else slice(h2_e, h2_s) if h2_e != 0 else slice(None)
            w1 = slice(w1_s, w1_e) if w1_s != 0 else slice(w1_e, w1_s) if w1_e != 0 else slice(None)
            w2 = slice(w2_s, w2_e) if w2_s != 0 else slice(w2_e, w2_s) if w2_e != 0 else slice(None)
            
            # Ensure that slices are valid
            try:
                input_patch1 = input_ycbcr[:, :, h1, w1]
                input_patch2 = input_ycbcr[:, :, h2, w2]
                output_patch1 = output[:, :, h1, w1]
                output_patch2 = output[:, :, h2, w2]
            except:
                continue
            
            # Calculate color difference weights
            color_diff = torch.sum((input_patch1 - input_patch2) ** 2, dim=1, keepdim=True)
            weight = torch.exp(color_diff * sigma_color)
            
            # Calculate gradient of the output
            grad = torch.norm(output_patch1 - output_patch2, p, dim=1, keepdim=True)
            
            # Accumulate the loss
            total_loss += torch.mean(weight * grad)
        
        return total_loss


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.l2_loss = nn.MSELoss()
        self.percep_loss = PercepLoss()
        self.smooth_loss = SmoothLoss()

    def forward(self, input, target):
        Fidelity_Loss = self.l2_loss(input, target)
        Percep_Loss = self.percep_loss(input, target)
        # Use target as guidance image for SmoothLoss, and input as output image
        Smooth_Loss = self.smooth_loss(target, input)
        return Fidelity_Loss, Percep_Loss, Smooth_Loss