import argparse
import os

import swanlab
import torch
import torch.nn as nn

import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from thop import profile, clever_format

from dataset import DIMDataset
from model.DIM import DIM


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

    # Calculate FLOPs using thop
    # Create a dummy input with shape (1, 3, 256, 256)
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Calculate FLOPs and parameters
    flops, params = profile(net, inputs=(dummy_input,), verbose=False)

    # record exp with swanlab
    run = swanlab.init(
        project="DIM-SCI",
        # 跟踪超参数与实验元数据
        config={
            "learning_rate": 1e-5,
            "epochs": args.n_iter,
            "loss_weight": {"L2": 1, "Percep": 0.2},
            "GPU": torch.cuda.current_device() if torch.cuda.is_available() else "cpu",
            "batch_size": 8,
            "dataset": "LOL-blur-selected",
            "seed": 123,
            "flops": int(flops),
            "params": int(params),
        },
    )

    # initialize the optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-5)

    # initialize best model tracking variables
    best_avg_loss = float('inf')
    best_model_path = None

    # run n_iter iterations of training
    for t in range(args.n_iter):
        net.train()
        # Track losses for this epoch
        losses = []
        L2_Losses = []
        Percep_Losses = []

        # get a single batch
        for (_, batch) in tqdm(enumerate(data_loader), desc=f'epoch-{t + 1}', total=len(data_loader)):
            # get the input and the target values
            x = batch[0].to(device)
            gt = batch[1].to(device)
            optimizer.zero_grad()
            
            # Total loss: L2 + Percep
            loss, L2_Loss, Percep_Loss = net._loss(x, gt)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5) 
            optimizer.step()

            # Record loss
            losses.append(loss.item())
            L2_Losses.append(L2_Loss.item())
            Percep_Losses.append(Percep_Loss.item())

        # Calculate average loss for this epoch
        avg_loss = np.mean(losses)
        run.log({"Loss": avg_loss, "L2_Loss": np.mean(L2_Losses), "Percep_Loss": np.mean(Percep_Losses)})

        # Save model with best (lowest) average loss
        if avg_loss < best_avg_loss:
            best_avg_loss = avg_loss
            best_model_path = os.path.join(args.result_path, 'best.pth')
            os.makedirs(args.result_path, exist_ok=True)
            torch.save(net.state_dict(), best_model_path)
            print(f'Best model saved at epoch {t + 1} with avg loss: {best_avg_loss}')

        # save each 5 epochs
        if (t + 1) % 5 == 0:
            model_path = os.path.join(args.result_path, f'epoch_{t + 1}.pth')
            os.makedirs(args.result_path, exist_ok=True)
            torch.save(net.state_dict(), model_path)
            print(f'Model saved at epoch {t + 1}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ReF-DIM Training')
    parser.add_argument('--n-iter', '-it', type=int, default=100)
    parser.add_argument('--data_path', '-d', type=str, default=r'path/to/dataset')
    parser.add_argument('--result_path', '-r', type=str, default=r'snapshot')
    args = parser.parse_args()

    train(args)
