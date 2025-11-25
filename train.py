import argparse
import os

import numpy as np
import swanlab
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DIMDataset
from model.DIM import DIM
from model.MyLoss import L_percep


def train(args):
    # set the random seeds for reproducibility
    np.random.seed(123)
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)

    # initialize the data loader
    data = DIMDataset(args.data_path)
    data_loader = DataLoader(data, batch_size=4, shuffle=True, pin_memory=True)

    # initialize the model and use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DIM().to(device)

    # record exp with swanlab
    run = swanlab.init(
        project="DIM",
        # 跟踪超参数与实验元数据
        config={
            "learning_rate": 1e-5,
            "epochs": args.n_iter,
            "loss_weight": {"L1_output": 1, "P": 1e-2},
            "GPU": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            "batch_size": 8,
            "dataset": "LOL-blur-selected",
            "seed": 123,
        },
    )

    # initialize the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    # initialize best model tracking variables
    best_avg_loss = float('inf')
    best_model_path = None

    # initialize loss functions
    pool = nn.AvgPool2d(8, 8).to(device)
    l1_loss = nn.MSELoss().to(device)
    low_loss = nn.L1Loss().to(device)
    p_loss = L_percep().to(device)

    # run n_iter iterations of training
    for t in range(args.n_iter):
        # Track losses for this epoch
        epoch_losses = []

        # get a single batch
        for (it, batch) in tqdm(enumerate(data_loader), desc=f'epoch-{t + 1}', total=len(data_loader)):
            # get the X and the targets values
            x = batch[0].to(device)
            gt = batch[1].to(device)

            # forward pass
            output, output_low = net(x)

            # L1 loss for output
            L1_loss = l1_loss(output, gt)
            Low_loss = low_loss(output_low, pool(gt))

            # Extract VGG features for L_exp loss
            # L_exp expects (input_feature, target_image) where input_feature is VGG feature
            P_loss = p_loss(output, gt)

            # Total loss: L1 + perceptual
            loss = L1_loss + 0.25 * Low_loss + 0.75 * P_loss

            if it % 8 == 0:
                run.log({
                    "L1 Loss Output": L1_loss.item(),
                    "Low Loss Output": Low_loss.item(),
                    "P Loss": P_loss.item(),
                    "Total Loss": loss.item()
                })

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Record loss
            epoch_losses.append(loss.item())

        # Calculate average loss for this epoch
        avg_loss = np.mean(epoch_losses)
        run.log({"Average Loss": avg_loss})

        # Save model with best (lowest) average loss
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_model_path = os.path.join(args.result_path, 'best.pth')
            os.makedirs(args.result_path, exist_ok=True)
            torch.save(net.state_dict(), best_model_path)
            print(f'Best model saved at epoch {t + 1} with avg loss: {best_avg_loss}')

    # print final best model info
    if best_model_path:
        print(f'Best model saved at {best_model_path} with avg loss: {best_avg_loss}')
    run.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReF-DIM Training')
    parser.add_argument('--n-iter', '-it', type=int, default=100)
    parser.add_argument('--data_path', '-d', type=str, default=r'path/to/dataset')
    parser.add_argument('--result_path', '-r', type=str, default=r'snapshot')
    args = parser.parse_args()

    train(args)
