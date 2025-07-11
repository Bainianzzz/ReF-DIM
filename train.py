import argparse
import os

import torch

import numpy as np
# from imageio.v2 import sizes
from torch.utils.data import DataLoader
from tqdm import tqdm

# from dataset import RegressionDataset
# from model import RegressionModel, RegressionTrain
from dataset import DIMDataset
from model.gradnorm import GradNorm
from model.ReF_DIM_C2f import ReF_DIM

import matplotlib.pyplot as plt


def train_toy_example(args):
    # set the random seeds for reproducibility
    np.random.seed(123)
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)

    # define the sigmas, the number of tasks and the epsilons
    # for the toy example
    # sigmas = [1.0, float(args.sigma)]
    # print('Training toy example with sigmas={}'.format(sigmas))
    n_tasks = args.n_tasks
    # epsilons = np.random.normal(scale=3.5, size=(n_tasks, 100, 250)).astype(np.float32)

    # initialize the data loader
    data = DIMDataset(args.data_path)
    data_loader = DataLoader(data, batch_size=8, shuffle=True, pin_memory=True)

    # initialize the model and use CUDA if available
    net = ReF_DIM(6)
    model = GradNorm(net)
    if torch.cuda.is_available():
        model.cuda()

    # initialize the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    n_iterations = int(args.n_iter)
    weights = []
    task_losses = []
    loss_ratios = []
    grad_norm_losses = []

    # run n_iter iterations of training
    for t in range(n_iterations):

        # get a single batch
        for (it, batch) in enumerate(tqdm(data_loader, desc=f'epoch-{t + 1}: ')):
            #  get the X and the targets values
            X = batch[0]
            ts = batch[1]
            if torch.cuda.is_available():
                X = X.cuda()
                ts = ts.cuda()

            # evaluate each task loss L_i(t)
            task_loss = model(X, ts)  # this will do a forward pass in the model and will also evaluate the loss
            # compute the weighted loss w_i(t) * L_i(t)
            weighted_task_loss = []
            for i in range(len(task_loss)):
                weighted_task_loss.append(torch.mul(model.weights, task_loss[i]))
                # initialize the initial loss L(0) if t=0
            task_loss = torch.stack(task_loss)
            weighted_task_loss = torch.stack(weighted_task_loss)
            if t == 0:
                mean = torch.mean(weighted_task_loss)
                # set L(0)
                if torch.cuda.is_available():
                    initial_weighted_task_loss = mean.data.cpu()
                else:
                    initial_weighted_task_loss = mean.data
                initial_weighted_task_loss = initial_weighted_task_loss.numpy()

            # get the total loss
            loss = torch.sum(weighted_task_loss)
            # clear the gradients
            optimizer.zero_grad()
            # do the backward pass to compute the gradients for the whole set of weights
            # This is equivalent to compute each \nabla_W L_i(t)
            loss.backward(retain_graph=True)

            # set the gradients of w_i(t) to zero because these gradients have to be updated using the GradNorm loss
            # print('Before turning to 0: {}'.format(model.weights.grad))
            model.weights.grad.data = model.weights.grad.data * 0.0
            # print('Turning to 0: {}'.format(model.weights.grad))

            # switch for each weighting algorithm:
            # --> grad norm
            if args.mode == 'grad_norm':

                # get layer of shared weights
                W = model.get_last_shared_layer()

                # get the gradient norms for each of the tasks
                # G^{(i)}_w(t) 
                norms = []
                for i in range(len(weighted_task_loss[0])):
                    try:
                        # get the gradient of this task loss with respect to the shared parameters
                        gygw = torch.autograd.grad(weighted_task_loss[i, :].sum(), W, retain_graph=True)
                        # compute the norm
                        norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
                    except IndexError:
                        pass
                norms = torch.stack(norms)
                # print('G_w(t): {}'.format(norms))

                # compute the inverse training rate r_i(t)
                # \curl{L}_i 
                if torch.cuda.is_available():
                    loss_ratio = weighted_task_loss.data.cpu().numpy() / initial_weighted_task_loss
                else:
                    loss_ratio = weighted_task_loss.data.numpy() / initial_weighted_task_loss
                # r_i(t)
                inverse_train_rate = loss_ratio / np.mean(loss_ratio)
                # print('r_i(t): {}'.format(inverse_train_rate))

                # compute the mean norm \tilde{G}_w(t)
                if torch.cuda.is_available():
                    mean_norm = np.mean(norms.data.cpu().numpy())
                else:
                    mean_norm = np.mean(norms.data.numpy())
                # print('tilde G_w(t): {}'.format(mean_norm))

                # compute the GradNorm loss
                # this term has to remain constant
                constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
                if torch.cuda.is_available():
                    constant_term = constant_term.cuda()
                # print('Constant term: {}'.format(constant_term))
                # this is the GradNorm loss itself
                try:
                    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                except RuntimeError:
                    constant_term = constant_term[:, :len(norms)]
                    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
                # print('GradNorm loss {}'.format(grad_norm_loss))

                # compute the gradient for the weights
                model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]

            # do a step with the optimizer
            optimizer.step()
            '''
            print('')
            wait = input("PRESS ENTER TO CONTINUE.")
            print('')
            '''

        # renormalize
        normalize_coeff = n_tasks / torch.sum(model.weights.data, dim=0)
        model.weights.data = model.weights.data * normalize_coeff

        # record
        if torch.cuda.is_available():
            task_losses.append(task_loss.data.cpu().numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.cpu().numpy())
            grad_norm_losses.append(grad_norm_loss.data.cpu().numpy())
        else:
            task_losses.append(task_loss.data.numpy())
            loss_ratios.append(np.sum(task_losses[-1] / task_losses[0]))
            weights.append(model.weights.data.numpy())
            grad_norm_losses.append(grad_norm_loss.data.numpy())

        if (t + 1) % 5 == 0:
            if torch.cuda.is_available():
                print('{}/{}: loss_ratio={}, weights={}, grad_norm_loss={}'.format(
                    t+1, args.n_iter, loss_ratios[-1], model.weights.data.cpu().numpy(),
                    grad_norm_loss.data.cpu().numpy()))
            # 保存模型参数
            if args.save_mode:
                if not os.path.exists(args.result_path):
                    os.mkdir(args.result_path)
                torch.save(net.state_dict(), f'{args.result_path}/model_gradnorm_{t}.pth')

            # else:
            #     print('{}/{}: loss_ratio={}, weights={}, task_loss={}, grad_norm_loss={}'.format(
            #         t, args.n_iter, loss_ratios[-1], model.weights.data.numpy(), task_loss.data.numpy(),
            #         grad_norm_loss.data.numpy()))

        torch.cuda.empty_cache()

    # task_losses = np.array(task_losses)
    # weights = np.array(weights)

    # plt.rc('text', usetex=True)
    # plt.rc('font', family='serif', size='smaller')
    # fig = plt.figure()
    # ax1 = fig.add_subplot(2, 3, 1)
    # ax1.set_title(r'Loss (scale $\sigma_0=1.0$)')
    # ax2 = fig.add_subplot(2, 3, 2)
    # ax2.set_title(r'Loss (scale $\sigma_1={})$'.format(sigmas[1]))
    # ax3 = fig.add_subplot(2, 3, 3)
    # ax3.set_title(r"$\sum_i L_i(t) / L_i(0)$")
    # ax4 = fig.add_subplot(2, 3, 4)
    # ax4.set_title(r'$L_{\text{grad}}$')
    #
    # ax5 = fig.add_subplot(2, 3, 5)
    # ax5.set_title(r'Change of weights $w_i$ over time')
    #
    # ax1.plot(task_losses[:, 0])
    # ax2.plot(task_losses[:, 1])
    # ax3.plot(loss_ratios)
    # ax4.plot(grad_norm_losses)
    # ax5.plot(weights[:, 0])
    # ax5.plot(weights[:, 1])
    # plt.show()

    # 绘制总损失 (total loss) 的折线图
    # plt.figure()
    # ax = plt.subplot(1, 1, 1)
    # ax.set_title('Total Loss over Training')
    #
    # # 假设 total_loss 是包含每个迭代步的总损失值的列表
    # # 这里使用 task_losses[:, 0] + task_losses[:, 1] 来计算每个迭代步的总损失
    # total_loss = np.sum(grad_norm_loss.detach().cpu().numpy(), axis=1)
    # ax.plot(total_loss)
    #
    # plt.xlabel('Iteration')
    # plt.ylabel('Total Loss')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GradNorm')
    parser.add_argument('--n-iter', '-it', type=int, default=100)
    parser.add_argument('--mode', '-m', choices=('grad_norm', 'equal_weight'), default='grad_norm')
    parser.add_argument('--alpha', '-a', type=float, default=0.12)
    parser.add_argument('--n_tasks', '-n', type=int, default=1)
    parser.add_argument('--data_path', '-d', type=str, default=r'D:\datasets\ReF_DIM')
    parser.add_argument('--result_path', '-r', type=str, default=r'result/snapshot')
    parser.add_argument('--save_mode', '-save', type=bool, default=True)
    args = parser.parse_args()

    train_toy_example(args)
