#!/usr/bin/env python3

import os
import sys
import glob
import argparse
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

torch.backends.cudnn.deterministic=True

from src import dataloader
from src import model

def train(args, device, net, train_loader, optimizer, epoch):
    net.train()
    loss_plot = 0
    for batch_idx, batch in enumerate(train_loader):
        for k, v in batch.items():
            batch[k] = v.to(device)
        flow_gt = batch['flow_gt']

        optimizer.zero_grad()
        flow_pred = net(batch)
        flow_x = flow_pred[:, 0]
        flow_y = flow_pred[:, 1]

        flow_loss_x = F.mse_loss(flow_x, flow_gt[:, 0], reduction='mean') 
        flow_loss_y = F.mse_loss(flow_y, flow_gt[:, 1], reduction='mean')
        loss = (flow_loss_x + flow_loss_y)/2.0
        loss_plot += loss.item()

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print(f"Train Epoch: {epoch} [({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {np.sqrt(loss.item()):.6f} flow_loss_x: {np.sqrt(flow_loss_x.item()):.6f}, flow_loss_y: {np.sqrt(flow_loss_y.item()):.6f}")
            if args.dry_run:
                break
    loss_plot /= len(train_loader)
    loss_plot = np.sqrt(loss_plot)
    return loss_plot

def test(net, device, test_loader):
    net.eval()

    test_loss_sum_mae = 0
    test_loss_sum_mse = 0

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)
            flow_gt = batch['flow_gt']
            altitude = batch['range'].clamp(0.1,)
            
            flow_pred = net(batch)
            flow_x = torch.arctan(flow_pred[:,0]/altitude[:,0])
            flow_y = torch.arctan(flow_pred[:,1]/altitude[:,0])
            flow_pred = torch.stack((flow_x, flow_y), -1)

            test_loss_sum_mae += F.l1_loss(flow_pred, flow_gt, reduction='mean').item()  
            test_loss_sum_mse += F.mse_loss(flow_pred, flow_gt, reduction='mean').item()

    test_loss_mae = test_loss_sum_mae/len(test_loader)
    test_loss_mse = test_loss_sum_mse/len(test_loader)
    print('\n[Test] L1 loss: {:.6f}, RMSE Loss: {:.6f}\n'.format(test_loss_mae, np.sqrt(test_loss_mse)))
    return np.sqrt(test_loss_mse)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train flow prediction model.')

    # Model setup.
    parser.add_argument('--model',     default='ResNet18', help="name of model in model.py")
    parser.add_argument('--use_range', action='store_true', default=False, help="whether using range for the training")

    # Dataset path.
    parser.add_argument('--train_dataset_path', 
                        help="Path to train dataset", required=True)
    parser.add_argument('--test_dataset_path', 
                        help="Path to test dataset", required=True)

    parser.add_argument('--model_save_path',
                        help="Dir to save model", required=True)

    # Training parameters.
    parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test_batch_size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry_run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log_interval', type=int, default=40, metavar='N',
                        help='how many batches to wait before logging training status')

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size, 'drop_last': True }
    test_kwargs  = {'batch_size': args.test_batch_size}
    model_kwargs = {'range_flag': args.use_range}

    if use_cuda:
        cuda_kwargs = {'num_workers': 4,
                       'shuffle': True, 
                       'worker_init_fn': lambda id: np.random.seed(id*args.seed),
                       }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Prepare the dataset.
    print("Loading dataset...")
    train_datasets = [dataloader.FlowDataset(path, transform=dataloader.FlipFlow()) for path in \
                      sorted(glob.glob(os.path.join(args.train_dataset_path, '*.npz')))]
    train_dataset = torch.utils.data.ConcatDataset(train_datasets)

    test_datasets = [dataloader.FlowDataset(path) for path in \
                      sorted(glob.glob(os.path.join(args.test_dataset_path, '*.npz')))]
    test_dataset = torch.utils.data.ConcatDataset(test_datasets)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    # Load network.
    if os.path.exists(args.model_save_path): 
        saved_model  = torch.load(args.model_save_path)
        assert saved_model['model_name'] == args.model
        model_kwargs.update(saved_model['model_kwargs'])
        state_dict   = saved_model['model_state_dict']
        net = getattr(model, args.model)(**model_kwargs).to(device)
        net.load_state_dict(state_dict)
        # sys.exit(0)
    else:
        net = getattr(model, args.model)(**model_kwargs).to(device)

    # Setup optimizer.
    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    # Begin training.
    train_loss_array = []
    test_loss_array = []

    least_test_loss = 1000
    for epoch in range(1, args.epochs + 1):
        train_loss = train(args, device, net, train_loader, optimizer, epoch)
        test_loss = test(net, device, test_loader)

        train_loss_array.append(train_loss)
        test_loss_array.append(test_loss)

        plt.plot(np.array(train_loss_array), 'b', label='Train Loss')
        plt.plot(np.array(test_loss_array), 'r', label='Test Loss')
        plt.scatter(np.argmin(np.array(test_loss_array)), np.min(test_loss_array), s=30, color='green')
        plt.title('Loss Plot, min:{:.3f}'.format(np.min(test_loss_array)))
        plt.legend()
        plt.grid()
        plt.ylim(0, 0.3)
        plt.savefig(os.path.splitext(args.model_save_path)[0] + '.jpg')
        plt.close()
        # scheduler.step()
        if test_loss < least_test_loss:
            least_test_loss = test_loss
            torch.save({'model_name': type(net).__name__,
                        'model_kwargs': model_kwargs,
                        'model_state_dict': net.state_dict(),
                        'epoch': epoch,
                        'test_loss': test_loss
                        }, args.model_save_path)
