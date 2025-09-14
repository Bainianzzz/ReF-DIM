import torch
from torch.xpu import device

from model.ReF_DIM_C2f import ReF_DIM
from model.MyLoss import *


class GradNorm(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

        self.weights = torch.nn.Parameter(torch.ones(model.n_task).float())

        self.l1Loss = nn.SmoothL1Loss()
        self.l_TV = L_TV()
        self.l_spa = L_spa()
        self.l_color = L_color()
        self.hist_loss = HistogramLoss(norm=True, reduction='sum')
        self.l_exp = L_exp(16, 0.6)

    def forward(self, input, target):
        output = self.model(input)

        batch_losses = []
        for i in range(input.shape[0]):
            losses = [self.l1Loss(target[i, :, :, :].unsqueeze(0), output[i, :, :, :].unsqueeze(0)),
                      self.l_TV(output[i, :, :, :].unsqueeze(0)),
                      torch.mean(self.l_exp(output[i, :, :, :].unsqueeze(0))),
                      self.l_spa(target[i, :, :, :].unsqueeze(0), output[i, :, :, :].unsqueeze(0)),
                      self.hist_loss(target[i, :, :, :].unsqueeze(0), output[i, :, :, :].unsqueeze(0)),
                      torch.mean(self.l_color(output[i, :, :, :].unsqueeze(0)))]
            batch_losses.append(torch.stack(losses))

        return batch_losses

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DIM_net = GradNorm(ReF_DIM(6)).to(device)
    x = torch.randn(1, 3, 256, 256).to(device)
    y = torch.randn(1, 3, 256, 256).to(device)
    print(DIM_net(x, y))