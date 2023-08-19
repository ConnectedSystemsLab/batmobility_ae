#!/usr/bin/env python3

"""This file contains the dataloader for the flow dataset.
"""

import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
torch.multiprocessing.set_sharing_strategy('file_system')

class FlipFlow(object):
    """ Data augmentation: flip flow and heatmap. """

    def __init__(self, channels=[0,1], probs=[0.5,0.5]):
        self.channels = channels
        self.probs = probs

    def __call__(self, sample):
        flow_gt = sample['flow_gt']
        heatmap = sample['radar1']

        for c, p in zip(self.channels, self.probs):
            if torch.rand(1).item() < p:
                flow_gt[c] *= -1
                heatmap[c,:,:] = transforms.functional.vflip(heatmap[c,:,:])

        sample['flow_gt'] = flow_gt
        sample['radar1'] = heatmap

        return sample

class FlowDataset(Dataset):
    """Flow dataset."""

    def __init__(self, path, transform=None):
        # Load files from .npz.
        self.path = path 
        with np.load(path) as data:
            self.files = data.files
            self.dataset = {k : data[k] for k in self.files} 

        # Check if lengths are the same.
        for k in self.files:
            print(k, self.dataset[k].shape, self.dataset[k].dtype)
        lengths = [self.dataset[k].shape[0] for k in self.files]
        assert len(set(lengths)) == 1
        self.num_samples = lengths[0] 

        # Save transforms.
        self.transform = transform
    
    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        sample = {k: torch.from_numpy(self.dataset[k][idx]) for k in self.files}
        if self.transform:
            sample = self.transform(sample)
        return sample

