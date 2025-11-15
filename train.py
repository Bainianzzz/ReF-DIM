import argparse
import os

import swanlab
import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DIMDataset
from model.DIM import DIM
from model.MyLoss import L_percep, L_edge
from utils import count_flops_and_params


def train(args):
    # set the random seeds for reproducibility
    np.random.seed(123)
    torch.cuda.manual_seed_all(123)
    torch.manual_seed(123)

    # initialize the data loader
    data = DIMDataset(args.data_path)
    data_loader = DataLoader(data, batch_size=8, shuffle=True, pin_memory=True)

    # initialize the model and use CUDA if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = DIM().to(device)

    # Calculate FLOPs and parameters
    flops, params = count_flops_and_params(net, input_shape=(1, 3, 224, 224), device=device)

    # record exp with swanlab
    run = swanlab.init(
        project="DIM",
        # 跟踪超参数与实验元数据
        config={
            "learning_rate": 2e-5,
            "epochs": args.n_iter,
            "loss_weight": {"L1_output": 1, "P": 1e-2, "Edge": 5},
            "GPU": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            "batch_size": 8,
            "dataset": "LOL-blur-selected",
            "seed": 123,
            "flops": flops,
            "params": params,
        },
    )

    # initialize the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=2e-5)

    # initialize best model tracking variables
    best_avg_loss = float('inf')
    best_model_path = None

    # initialize loss functions
    l1_loss = nn.L1Loss()
    p_loss = L_percep().to(device)
    edge_loss = L_edge().to(device)

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
            output = net(x)
            
            # L1 loss for output
            L1_loss_output = l1_loss(output, gt)
            
            # Extract VGG features for L_exp loss
            # L_exp expects (input_feature, target_image) where input_feature is VGG feature
            P_loss = p_loss(output, gt)
            
            # Edge/TV loss for smoothness (applied to output)
            Edge_loss = edge_loss(output)
            
            # Total loss: L1 + perceptual + edge
            loss = L1_loss_output + 1e-2 * P_loss + 50 * Edge_loss
            
            if it % 8 == 0:
                run.log({
                    "L1 Loss Output": L1_loss_output.item(),
                    "Exp Loss": P_loss.item(),
                    "Edge Loss": Edge_loss.item(),
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
