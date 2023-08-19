#!/usr/bin/env python3

import os
import glob
import argparse
import torch
import numpy as np
np.set_printoptions(precision=3, floatmode='fixed', sign=' ')
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 22})
import matplotlib.pyplot as plt

from src import model
from src.dataloader import FlowDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',  
                        help="Path to data directory.",
                        required=True)
    parser.add_argument('--saved_model_path',
                        help="Path to saved model.",
                        required=True)
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # device = "cpu"

    # Load Trained NN
    saved_model  = torch.load(args.saved_model_path)
    model_name   = saved_model['model_name']
    model_kwargs = saved_model['model_kwargs']
    # model_kwargs.update({'range_flag': False})
    state_dict   = saved_model['model_state_dict']
    net = getattr(model, model_name)(**model_kwargs).to(device)
    net.load_state_dict(state_dict)
    net.eval()

    # Get list of bag files in root directory.
    npz_paths = sorted(glob.glob(os.path.join(args.data_dir, '*.npz')))
    npz_paths = [file for file in npz_paths if not file.endswith('test.npz')]

    mean_optflow_mae = []
    mean_pred_mae = []
    mean_optflow_rmse = []
    mean_pred_rmse = []

    for path in npz_paths:
        print(f"Processing {path}...")

        dataset = FlowDataset(path)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=0)

        optflow_xs, optflow_ys = [], []
        flow_pred_xs, flow_pred_ys = [], []
        flow_gt_xs, flow_gt_ys = [], []
        altitudes, altitudes_gt = [], []

        for i, batch in enumerate(test_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            optflow = batch['flow'].cpu()
            flow_gt  = batch['flow_gt'].cpu()
            altitude_gt = batch['range_gt'].cpu().numpy()
            altitude = batch['range'].clamp(0.1,).cpu().numpy()

            optflow_xs.append(optflow[:,0])
            optflow_ys.append(-optflow[:,1])

            with torch.no_grad():
                flow_pred = net(batch).cpu()

            # flow_x, flow_y = flow_pred[:,0], flow_pred[:,1]
            flow_pred = torch.arctan(flow_pred/altitude[:,0])

            flow_pred_x, flow_pred_y = flow_pred[:,0], flow_pred[:,1]
            flow_pred_x, flow_pred_y = -flow_pred_y, flow_pred_x
            flow_pred_xs.append(flow_pred_x)
            flow_pred_ys.append(flow_pred_y)

            flow_gt_x, flow_gt_y = flow_gt[:,0], flow_gt[:,1]
            flow_gt_xs.append(flow_gt_x)
            flow_gt_ys.append(flow_gt_y)

            altitudes.append(altitude[:,0])
            altitudes_gt.append(altitude_gt[:,0])

        optflow_xs, optflow_ys = np.array(optflow_xs), np.array(optflow_ys)
        flow_pred_xs, flow_pred_ys = np.array(flow_pred_xs), np.array(flow_pred_ys)
        flow_gt_xs, flow_gt_ys = np.array(flow_gt_xs), np.array(flow_gt_ys)
        altitudes, altitudes_gt = np.array(altitudes), np.array(altitudes_gt)

        pred_mae = (np.mean(np.abs(flow_pred_xs - flow_gt_xs))+np.mean(np.abs(flow_pred_ys - flow_gt_ys)))/2
        optflow_mae = (np.mean(np.abs(optflow_xs - flow_gt_xs))+np.mean(np.abs(optflow_ys - flow_gt_ys)))/2
        print(f"Pred MAE: {pred_mae:.3f}")
        print(f"Optflow MAE: {optflow_mae:.3f}")

        pred_rmse = (np.sqrt(np.mean((flow_pred_xs - flow_gt_xs)**2))+np.sqrt(np.mean((flow_pred_ys - flow_gt_ys)**2)))/2
        optflow_rmse = (np.sqrt(np.mean((optflow_xs - flow_gt_xs)**2))+np.sqrt(np.mean((optflow_ys - flow_gt_ys)**2)))/2
        print(f"Pred RMSE: {pred_rmse:.3f}")
        print(f"Optflow RMSE: {optflow_rmse:.3f}")

        mean_optflow_mae.append(optflow_mae)
        mean_optflow_rmse.append(optflow_rmse)

        mean_pred_mae.append(pred_mae)
        mean_pred_rmse.append(pred_rmse)

        fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10,5))

        # ax[0].set_title(f"Pred MAE: {pred_mae:.3f} Pred RMSE: {pred_rmse:.3f}")
        ax[0].plot(optflow_xs,            color='r', label='Optflow')
        ax[0].plot(flow_gt_xs,            color='g', label='Ground Truth')
        ax[0].plot(flow_pred_xs,          color='b', label='Pred')
        ax[0].set_ylim(-1,1)
        ax[0].set_ylabel('$\omega_x$ rad/s')

        # ax[1].set_title(f"Optflow MAE: {optflow_mae:.3f} Optflow RMSE: {optflow_rmse:.3f}")
        ax[1].plot(optflow_ys,            color='r')
        ax[1].plot(flow_gt_ys,            color='g')
        ax[1].plot(flow_pred_ys,          color='b')
        ax[1].set_ylim(-1,1)
        ax[1].set_ylabel('$\omega_y$ (rad/s)')

        # fig.tight_layout()
        fig.legend()
        fig.savefig(f'{os.path.splitext(args.saved_model_path)[0]}_{os.path.basename(os.path.splitext(path)[0])}_test.jpg')
        plt.close(fig)

        d = {'optflow_xs': optflow_xs, 
             'optflow_ys': optflow_ys, 
             'flow_pred_xs': flow_pred_xs,
             'flow_pred_ys': flow_pred_ys,
             'flow_gt_xs': flow_gt_xs,
             'flow_gt_ys': flow_gt_ys,
             'altitudes': altitudes,
             'altitudes_gt': altitudes_gt}
        np.savez(f'{os.path.splitext(args.saved_model_path)[0]}_{os.path.basename(os.path.splitext(path)[0])}_test.npz', **d)
        
    print(f"pred mae total {np.mean(mean_pred_mae):.3f} pred rmse total {np.mean(mean_pred_rmse):.3f}")
    print(f"optflow mae total {np.mean(mean_optflow_mae):.3f} optflow rmse total {np.mean(mean_optflow_rmse):.3f}")
