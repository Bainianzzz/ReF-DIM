import torch

from model.ReF_DIM_C2f import ReF_DIM
from model.MyLoss import *


class GradNorm(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_num = model.n_task

        self.weights = torch.nn.Parameter(torch.ones(self.loss_num).float())

        self.l1Loss = nn.SmoothL1Loss()
        self.l_TV = L_TV()
        self.l_spa = L_spa()
        self.l_color = L_color()
        self.hist_loss = HistogramLoss(norm=True, reduction='sum')
        self.l_exp = L_exp(16, 0.6)

    def forward(self, input, target):
        output = self.model(input)

        assert (self.loss_num >= 4)
        batch_losses = []
        for i in range(input.shape[0]):
            # if target[i, :, :, :].mean() == 0:
            #     losses[0]+=self.l_TV(output[i, :, :, :].unsqueeze(0))
            #     losses[1]+=torch.mean(self.l_spa(output[i, :, :, :].unsqueeze(0), input[i, :, :, :].unsqueeze(0)))
            #     losses[2]+=torch.mean(self.l_color(output[i, :, :, :].unsqueeze(0)))
            #     losses[3]+=torch.mean(self.l_exp(output[i, :, :, :].unsqueeze(0)))
            # else:
            losses = [self.l1Loss(target[i, :, :, :].unsqueeze(0), output[i, :, :, :].unsqueeze(0)),
                      self.l_TV(output[i, :, :, :].unsqueeze(0)),
                      torch.mean(self.l_exp(output[i, :, :, :].unsqueeze(0))),
                      self.l_spa(target[i, :, :, :].unsqueeze(0), output[i, :, :, :].unsqueeze(0)),
                      self.hist_loss(target[i, :, :, :].unsqueeze(0), output[i, :, :, :].unsqueeze(0)),
                      torch.mean(self.l_color(output[i, :, :, :].unsqueeze(0)))]
            batch_losses.append(torch.stack(losses))

        return batch_losses

    def get_last_shared_layer(self):
        return self.weights


class RegressionTrain(torch.nn.Module):
    '''
    '''

    def __init__(self, model):
        '''
        '''

        # initialize the module using super() constructor
        super(RegressionTrain, self).__init__()
        # assign the architectures
        self.model = model
        # assign the weights for each task
        self.weights = torch.nn.Parameter(torch.ones(model.n_tasks).float())
        # loss function
        self.mse_loss = nn.MSELoss()

    def forward(self, x, ts):
        B, n_tasks = ts.shape[:2]
        ys = self.model(x)

        # check if the number of tasks is equal to this size
        assert (ys.size()[1] == n_tasks)
        task_loss = []
        for i in range(n_tasks):
            task_loss.append(self.mse_loss(ys[:, i, :], ts[:, i, :]))
        task_loss = torch.stack(task_loss)

        return task_loss

    def get_last_shared_layer(self):
        return self.model.get_last_shared_layer()


class RegressionModel(torch.nn.Module):
    '''
    '''

    def __init__(self, n_tasks):
        '''
        Constructor of the architecture.
        Input:
            n_tasks: number of tasks to solve ($T$ in the paper)
        '''

        # initialize the module using super() constructor
        super(RegressionModel, self).__init__()

        # number of tasks to solve
        self.n_tasks = n_tasks
        # fully connected layers
        self.l1 = torch.nn.Linear(250, 100)
        self.l2 = torch.nn.Linear(100, 100)
        self.l3 = torch.nn.Linear(100, 100)
        self.l4 = torch.nn.Linear(100, 100)
        # branches for each task
        for i in range(self.n_tasks):
            setattr(self, 'task_{}'.format(i), torch.nn.Linear(100, 100))

    def forward(self, x):
        # forward pass through the common fully connected layers
        h = F.relu(self.l1(x))
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        h = F.relu(self.l4(h))

        # forward pass through each output layer
        outs = []
        for i in range(self.n_tasks):
            layer = getattr(self, 'task_{}'.format(i))
            outs.append(layer(h))

        return torch.stack(outs, dim=1)

    def get_last_shared_layer(self):
        return self.l4


if __name__ == '__main__':
    DIM_net = GradNorm(ReF_DIM(4)).cuda()
    x = torch.randn(1, 3, 256, 256).cuda()
    y = torch.randn(1, 3, 256, 256).cuda()
    print(DIM_net(x, y).size())

    net = RegressionTrain(RegressionModel(4)).cuda()
    x = torch.randn(1, 250).cuda()
    y = torch.randn(1, 4, 100).cuda()
    print(net(x, y).size())
