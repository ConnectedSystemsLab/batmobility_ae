#!/usr/bin/env python3

import os
import sys
import glob
import subprocess
import multiprocessing

test_paths       = ['test_data/v4',
                    'test_data/v5',
                    'test_data/v6']
model_save_paths = ['test_data/v4/v4_2_6x60x3_24x24_10.0x25.0_ResNet18Mini.pth',
                    'test_data/v5/v5_3_3x60x4_24x24_10.0x25.0_ResNet18Micro.pth',
                    'test_data/v6/v6_3_6x60x6_24x24_10.0x25.0_ResNet18Micro.pth']
radar_buffer_lens         = [2, 3, 3]
angle_ress                = [6, 3, 6]
angle_ranges              = [60, 60, 60]
range_subsampling_factors = [3, 4, 6]
normalization_ranges      = [[10.0, 25.0], [10.0, 25.0], [10.0, 25.0]]
resize_shapes             = [[24, 24], [24, 24], [24, 24]]

def create_dataset(args):
   return create_dataset_(*args)

def create_dataset_(bag_path, 
                    radar_buffer_len, 
                    angle_res, angle_range, 
                    range_subsampling_factor, 
                    normalization_range, 
                    resize_shape):

    cmd = [sys.executable, 
           "create_dataset_test.py", 
           f"--bag_path={bag_path}",
           f"--radar_buffer_len={radar_buffer_len}",
           f"--angle_res={angle_res}",
           f"--angle_range={angle_range}",
           f"--range_subsampling_factor={range_subsampling_factor}",
           f"--normalization_range", f"{normalization_range[0]}", f"{normalization_range[1]}",
           f"--resize_shape",  f"{resize_shape[0]}", f"{resize_shape[1]}"]
    print(cmd)
    subprocess.run(cmd, check=True)

def eval_model(model_save_path, 
               test_path):

    cmd = [sys.executable,
           "test.py",
           f"--data_dir={test_path}",
           f"--saved_model_path={model_save_path}"]
    print(cmd)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    
    with open(os.path.splitext(model_save_path)[0] + '.txt', 'w') as f:
        f.writelines(result.stdout)

if __name__ == '__main__':

    for x in zip(test_paths,                                                                                                                             model_save_paths, 
                 radar_buffer_lens, 
                 angle_ress, 
                 angle_ranges, 
                 range_subsampling_factors, 
                 normalization_ranges, 
                 resize_shapes):

        eval_path = x[0]
        model_save_path = x[1]
        radar_buffer_len = x[2]
        angle_res = x[3]
        angle_range = x[4]
        range_subsampling_factor = x[5]
        normalization_range = x[6]
        resize_shape= x[7]

        # Create test dataset from all bags in test_path
        eval_bag_paths = sorted(glob.glob(os.path.join(eval_path, '*.bag')))

        p = multiprocessing.Pool(4)
        p.map(create_dataset, [(bag_path, 
                                radar_buffer_len, 
                                angle_res, angle_range, 
                                range_subsampling_factor, 
                                normalization_range, 
                                resize_shape) for bag_path in eval_bag_paths])

        # Evaluate model on test dataset
        eval_model(model_save_path, eval_path)

