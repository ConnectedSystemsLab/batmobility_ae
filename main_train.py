#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
import multiprocessing

configs                   = ['v6', 'v5', 'v4']
radar_buffer_lens         = ['3',  '3',  '2']
angle_ress                = ['6',  '3',  '6']
angle_ranges              = ['60', '60', '60']
range_subsampling_factors = ['6',  '4',  '3']
normalization_ranges      = [['10.0', '25.0'], ['10.0', '25.0'], ['10.0', '25.0']]
resize_shapes             = [['24', '24'], ['24', '24'], ['24', '24']]
models                    = ['ResNet18Micro', 'ResNet18Micro', 'ResNet18']
batch_size = 128
lr         = 1e-4
epochs     = 150

def create_dataset(args):
   return create_dataset_(*args)

def create_dataset_(bag_path, 
                    radar_buffer_len, 
                    angle_res, angle_range, 
                    range_subsampling_factor, 
                    normalization_range, 
                    resize_shape):

    cmd = [sys.executable, 
           "create_dataset.py", 
           f"--bag_path={bag_path}",
           f"--radar_buffer_len={radar_buffer_len}",
           f"--angle_res={angle_res}",
           f"--angle_range={angle_range}",
           f"--range_subsampling_factor={range_subsampling_factor}",
           f"--normalization_range", f"{normalization_range[0]}", f"{normalization_range[1]}",
           f"--resize_shape",  f"{resize_shape[0]}", f"{resize_shape[1]}"]
    print(cmd)
    subprocess.run(cmd, check=True)

def train_model(model, 
                train_path, 
                test_path, 
                output_dir, 
                batch_size, 
                lr, 
                epochs):

    model_name = f"{model}"
    model_save_path = os.path.join(output_dir, output_dir+f'_{model}') + '.pth'

    cmd = [sys.executable, 
           "train.py", 
           f"--model={model_name}",
           f"--use_range",
           f"--train_dataset_path={train_path}",
           f"--test_dataset_path={test_path}",
           f"--model_save_path={model_save_path}",
           f"--batch_size={batch_size}",
           f"--lr={lr}",
           f"--epochs={epochs}"]
    print(cmd)
    subprocess.run(cmd, check=True)

    cmd = [sys.executable,
           "validate.py",
           f"--data_dir={test_path}",
           f"--saved_model_path={model_save_path}"]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    with open(os.path.splitext(model_save_path)[0] + '.txt', 'w') as f:
        f.writelines(result.stdout)


if __name__ == '__main__':

    for x in zip(configs, 
                 radar_buffer_lens, 
                 angle_ress, 
                 angle_ranges, 
                 range_subsampling_factors, 
                 normalization_ranges, 
                 resize_shapes,
                 models):

        config = x[0]
        radar_buffer_len = x[1]
        angle_res = x[2]
        angle_range = x[3]
        range_subsampling_factor = x[4]
        normalization_range = x[5]
        resize_shape = x[6]
        model = x[7]

        # Create train and test datasets.
        train_path = os.path.join('train_data', config, 'train')
        train_bag_paths = sorted(glob.glob(os.path.join(train_path, '*.bag')))
        test_path = os.path.join('train_data', config, 'test')
        test_bag_paths = sorted(glob.glob(os.path.join(test_path, '*.bag')))
        bag_paths = train_bag_paths + test_bag_paths

        p = multiprocessing.Pool(4)
        p.map(create_dataset, [(bag_path, 
                                radar_buffer_len, 
                                angle_res, 
                                angle_range, 
                                range_subsampling_factor, 
                                normalization_range, 
                                resize_shape) for bag_path in bag_paths])

        # Create output directory.
        output_dir = f"{config}_{radar_buffer_len}_{angle_res}x{angle_range}x{range_subsampling_factor}_{resize_shape[0]}x{resize_shape[1]}_{normalization_range[0]}x{normalization_range[1]}"
        os.makedirs(output_dir, exist_ok=True)

        # Train models.
        train_model(model, train_path, test_path, output_dir,  batch_size, lr, epochs)

