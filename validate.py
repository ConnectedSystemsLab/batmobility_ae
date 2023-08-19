#!/usr/bin/env python3

import os
import glob
import argparse
import torch
from scipy.spatial.transform import Rotation as R
import numpy as np
np.set_printoptions(precision=3, floatmode='fixed', sign=' ')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    npz_paths = [file for file in npz_paths if not file.endswith('val.npz')]

    mean_pred_mae = []
    mean_pred_rmse = []

    for path in tqdm(npz_paths):
        print(f"Processing {path}...")

        dataset = FlowDataset(path)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=1,
                                                  shuffle=False, num_workers=0)

        flow_pred_xs, flow_pred_ys = [], []
        flow_gt_xs, flow_gt_ys = [], []
        altitudes, altitudes_gt = [], []

        for i, batch in enumerate(test_loader):
            for k, v in batch.items():
                batch[k] = v.to(device)
            flow_gt  = batch['flow_gt'].cpu()
            altitude_gt = batch['range_gt'].cpu().numpy()
            altitude = batch['range'].clamp(0.1,).cpu().numpy()

            with torch.no_grad():
                flow_pred = net(batch).cpu()

            # flow_x, flow_y = flow_pred[:,0], flow_pred[:,1]
            flow_pred = torch.arctan(flow_pred/altitude[:,0])

            flow_x, flow_y = flow_pred[:,0], flow_pred[:,1]
            flow_pred_xs.append(flow_x)
            flow_pred_ys.append(flow_y)

            flow_gt_x, flow_gt_y = flow_gt[:,0], flow_gt[:,1]
            flow_gt_xs.append(flow_gt_x)
            flow_gt_ys.append(flow_gt_y)

            altitudes.append(altitude[:,0])
            altitudes_gt.append(altitude_gt[:,0])

        flow_pred_xs, flow_pred_ys = np.array(flow_pred_xs), np.array(flow_pred_ys)
        flow_gt_xs, flow_gt_ys = np.array(flow_gt_xs), np.array(flow_gt_ys)
        altitudes, altitudes_gt = np.array(altitudes), np.array(altitudes_gt)

        print(f"MAE x: {np.mean(np.abs(flow_pred_xs - flow_gt_xs)):.3f}")
        print(f"MAE y: {np.mean(np.abs(flow_pred_ys - flow_gt_ys)):.3f}")

        print(f"RMSE x: {np.sqrt(np.mean((flow_pred_xs - flow_gt_xs)**2)):.3f}")
        print(f"RMSE y: {np.sqrt(np.mean((flow_pred_ys - flow_gt_ys)**2)):.3f}")

        print(f"err_mean x: {np.mean((flow_pred_xs - flow_gt_xs)):.3f}")
        print(f"err_std x:  {np.std((flow_pred_xs - flow_gt_xs)):.3f}")

        print(f"err_mean y: {np.mean((flow_pred_ys - flow_gt_ys)):.3f}")
        print(f"err_std y:  {np.std((flow_pred_ys - flow_gt_ys)):.3f}")

        pred_mae = (np.mean(np.abs(flow_pred_xs - flow_gt_xs))+np.mean(np.abs(flow_pred_ys - flow_gt_ys)))/2
        pred_rmse = (np.sqrt(np.mean((flow_pred_xs - flow_gt_xs)**2))+np.sqrt(np.mean((flow_pred_ys - flow_gt_ys)**2)))/2

        mean_pred_mae.append(pred_mae)
        mean_pred_rmse.append(pred_rmse)

        fig, ax = plt.subplots(5, 1, sharex=True, figsize=(5,10))

        ax[0].set_title(f"MAE x: {np.mean(np.abs(flow_pred_xs - flow_gt_xs)):.3f} RMSE x: {np.sqrt(np.mean((flow_pred_xs - flow_gt_xs)**2)):.3f}")
        ax[0].plot(flow_gt_xs,            label='flow_gt_x', color='b')
        ax[0].plot(flow_pred_xs,               label='flow_x', color='r')
        ax[0].set_ylim(-1,1)

        ax[1].set_title(f"err mean x: {np.mean((flow_pred_xs - flow_gt_xs)):.3f} stdev: {np.std((flow_pred_xs - flow_gt_xs)):.3f}")
        ax[1].plot(flow_pred_xs-flow_gt_xs,    label='err_x', color='g')
        ax[1].set_ylim(-.1,.1)

        ax[2].set_title(f"MAE y: {np.mean(np.abs(flow_pred_ys - flow_gt_ys)):.3f} RMSE y: {np.sqrt(np.mean((flow_pred_ys - flow_gt_ys)**2)):.3f}")
        ax[2].plot(flow_gt_ys,            label='flow_gt_y', color='b')
        ax[2].plot(flow_pred_ys,               label='flow_y', color='r')
        ax[2].set_ylim(-1,1)


        ax[3].set_title(f"err mean y: {np.mean((flow_pred_ys - flow_gt_ys)):.3f} stdev: {np.std((flow_pred_ys - flow_gt_ys)):.3f}")
        ax[3].plot(flow_pred_ys-flow_gt_ys,    label='err_y', color='g')
        ax[3].set_ylim(-.1,.1)

        ax[4].plot(altitudes_gt, label='altitude_gt', color='b')
        ax[4].plot(altitudes,    label='altitude', color='r')
        ax[4].set_ylim(0,np.max(altitudes_gt))

        fig.tight_layout()
        fig.legend()
        fig.savefig(f'{os.path.splitext(args.saved_model_path)[0]}_{os.path.basename(os.path.splitext(path)[0])}_val.jpg')
        plt.close(fig)

        d = {'flow_pred_xs': flow_pred_xs,
             'flow_pred_ys': flow_pred_ys,
             'flow_gt_xs': flow_gt_xs,
             'flow_gt_ys': flow_gt_ys,
             'altitudes': altitudes,
             'altitudes_gt': altitudes_gt}
        np.savez(f'{os.path.splitext(args.saved_model_path)[0]}_{os.path.basename(os.path.splitext(path)[0])}_val.npz', **d)

    print(f"pred mae total {np.mean(mean_pred_mae):.3f} pred rmse total {np.mean(mean_pred_rmse):.3f}")




    
