#!/usr/bin/env python3

"""
This script is to get the [heatmaps, flows] dataset from the .bag files for testing purposes.
"""

import os
import argparse
import rosbag
import cv2
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import numpy as np
from collections import defaultdict
from collections import deque
import matplotlib
matplotlib.use('TkAgg')

np.set_printoptions(precision=3, floatmode='fixed', sign=' ')

from src import dsp

def preprocess_2d_radar_6843ods(radar_cube,
                                angle_res=1, angle_range=90, 
                                range_subsampling_factor=2,
                                min_val=10.0, max_val=None,
                                resize_shape=(48,48)):
    """ Turn radar cube into x and y doppler-angle heatmaps. Assumes the following 
        antenna array layout (for xWR6843ISK-ODS):
                                    +---+---+---+---+
                                    | 1 | 4 | 5 | 8 |
                                    +---+---+---+---+
                                    | 2 | 3 | 6 | 7 |
                                    +---+---+---+---+
                                            | 9 |12 |
                                            +---+---+
                                            | 10| 11|
                                            +---+---+
    Args:
        radar_cube: 3D numpy array of shape (num_samples, num_antennas, num_chirps)
        angle_res: angle resolution of the resulting heatmap in degrees
        angle_range: angle range of the resulting heatmap in degrees
        range_subsampling_factor: subsampling factor for range bins.
        min_val: minimum value for normalization
        max_val: maximum value for normalization
        resize_shape: shape of the final heatmap

    Returns:
        2D numpy array of shape (2, resize_shape[0], resize_shape[1])
    """

    x_cube1 = np.stack([radar_cube[:,0,:],
                        radar_cube[:,3,:],
                        radar_cube[:,4,:],
                        radar_cube[:,7,:]], axis=1)
    x_cube2 = np.stack([radar_cube[:,1,:],
                        radar_cube[:,2,:],
                        radar_cube[:,5,:],
                        radar_cube[:,6,:]], axis=1)
    x_cube = x_cube1 + x_cube2

    y_cube1 = np.stack([radar_cube[:,4,:],
                        radar_cube[:,5,:],
                        radar_cube[:,8,:],
                        radar_cube[:,9,:]], axis=1)
    y_cube2 = np.stack([radar_cube[:,7,:],
                        radar_cube[:,6,:],
                        radar_cube[:,11,:],
                        radar_cube[:,10,:]], axis=1)
    y_cube = y_cube1 + y_cube2

    x_heatmap = dsp.compute_doppler_angle(x_cube, angle_res, angle_range, 
                                          range_subsampling_factor=range_subsampling_factor)
    y_heatmap = dsp.compute_doppler_angle(y_cube, angle_res, angle_range,
                                          range_subsampling_factor=range_subsampling_factor)

    x_heatmap = dsp.normalize(x_heatmap, min_val=min_val, max_val=max_val)
    y_heatmap = dsp.normalize(y_heatmap, min_val=min_val, max_val=max_val)

    x_heatmap = cv2.resize(x_heatmap, resize_shape, interpolation=cv2.INTER_AREA)
    y_heatmap = cv2.resize(y_heatmap, resize_shape, interpolation=cv2.INTER_AREA)

    return np.stack((x_heatmap, y_heatmap), axis=0)

def unpack_bag(bag, 
               radar_buffer_len=2,
               angle_res=1, 
               angle_range=90, 
               range_subsampling_factor=2,
               normalization_range=(10.0, 25.0),
               resize_shape=(48,48)):
    """ Preprocesses data inside a .bag file into test dataset.
    """

    flow_ts, flow_msgs         = [], []
    flow_gt_ts, flow_gt_msgs   = [], []
    range_ts, range_msgs       = [], []
    range_gt_ts, range_gt_msgs = [], []
    radar1_ts, radar1_msgs     = [], []

    radar1_buffer = deque(maxlen=radar_buffer_len)

    last_tracking_ts = None

    last_gt_ts = None
    last_position = np.zeros(3)
    last_rotation = np.zeros((3,3))
    last_gt_range = None

    for i, (topic, msg, ts) in tqdm(enumerate(bag.read_messages(['/mavros/distance_sensor/hrlv_ez4_pub',
                                                                 '/mavros/px4flow/raw/optical_flow_rad',
                                                                 '/tracking/odom/sample',
                                                                 '/radar1/radar_data'])), total=bag.get_message_count()):

        # Get range ground truth.
        if topic == '/mavros/distance_sensor/hrlv_ez4_pub':
            current_range = msg.range
            # print(f'range: {current_range:.3f}')
            if current_range > 0:
                last_gt_range = current_range 

            range_gt_msgs.append(np.array([last_gt_range], dtype=np.float32))
            range_gt_ts.append(ts.secs + 1e-9*ts.nsecs)

        # Get optical flow from optical flow sensor.
        if topic == '/mavros/px4flow/raw/optical_flow_rad':
            integration_time = msg.integration_time_us*1e-6
            flow_x = msg.integrated_x/integration_time
            flow_y = msg.integrated_y/integration_time
            quality = msg.quality

            flow_msgs.append(np.array([flow_x, flow_y], dtype=np.float32))
            flow_ts.append(ts.secs + 1e-9*ts.nsecs)
            # print(f'flow_x: {flow_x:9< .2f} flow_y: {flow_y:9< .2f}')
        
        # Get ground truth optical flow.
        if topic == '/tracking/odom/sample':
            curr_ts = ts.secs + 1e-9*ts.nsecs
            # Downsample to 30fps.
            if last_tracking_ts is None:
                last_tracking_ts = curr_ts 
                continue
            else:
                ts_diff = curr_ts - last_tracking_ts
                if ts_diff < 33e-3:
                    continue
                else:
                    last_tracking_ts = curr_ts 

            if last_gt_range is None:
                continue

            pose = msg.pose.pose
            position = np.array([pose.position.x, 
                                 pose.position.y, 
                                 pose.position.z])
            rotation = R.from_quat([pose.orientation.x, 
                                    pose.orientation.y, 
                                    pose.orientation.z, 
                                    pose.orientation.w]).as_matrix()

            if last_gt_ts is None:
                last_gt_ts = curr_ts 
                last_position = position
                last_rotation = rotation
                continue

            elapsed = curr_ts - last_gt_ts

            relative_position = position - last_position
            relative_pose = last_rotation.T @ relative_position

            flow_x, flow_y = np.arctan2([relative_pose[1], relative_pose[0]], last_gt_range)/elapsed 

            flow_gt_msgs.append(np.array([flow_x, flow_y], dtype=np.float32))
            flow_gt_ts.append(curr_ts)

            last_gt_ts = curr_ts 
            last_position = position
            last_rotation = rotation

        # Get radar data.
        if topic == '/radar1/radar_data':  # bot
            if last_gt_range is None:
                continue

            # 2d radar is used.
            radar_cube = dsp.reshape_frame(msg, flip_ods_phase=True)

            # Get current altitude.
            r = dsp.compute_altitude(radar_cube, 
                                     msg.range_max/msg.shape[2],
                                     msg.range_bias)
            range_msgs.append(np.array([r], dtype=np.float32))
            range_ts.append(ts.secs + 1e-9*ts.nsecs)

            # Accumulate radar cubes in buffer.
            radar1_buffer.append(radar_cube)
            if len(radar1_buffer) < radar1_buffer.maxlen:
                continue
            radar_cube = np.concatenate(radar1_buffer, axis=0)

            # Create heatmap(s).
            heatmap = preprocess_2d_radar_6843ods(radar_cube,
                                                  angle_res=angle_res, 
                                                  angle_range=angle_range, 
                                                  range_subsampling_factor=range_subsampling_factor,
                                                  min_val=normalization_range[0], max_val=normalization_range[1],
                                                  resize_shape=resize_shape) 

            radar1_msgs.append(heatmap)
            radar1_ts.append(ts.secs + 1e-9*ts.nsecs)

    flow_ts     = np.array(flow_ts)           # flow estimate from optical flow sensor 
    range_ts    = np.array(range_ts)          # range estimate from radar
    range_gt_ts = np.array(range_gt_ts)       # range ground truth from lidar/camera
    flow_gt_ts  = np.array(flow_gt_ts)        # flow ground truth from VIO
    radar1_ts   = np.array(radar1_ts)         # downward facing radar data

    d = defaultdict(lambda : [])

    if len(flow_ts) > 0:
        d['flow']       = [flow_ts, flow_msgs]
    if len(range_ts) > 0:
        d['range']      = [range_ts, range_msgs]
    if len(range_gt_ts) > 0:
        d['range_gt']   = [range_gt_ts, range_gt_msgs]
    if len(flow_gt_ts) > 0:
        d['flow_gt']    = [flow_gt_ts, flow_gt_msgs] 
    if len(radar1_ts) > 0:
        d['radar1']     = [radar1_ts, radar1_msgs]

    return d

def sync2topic(unpacked_bag, sync_topic):
    """ Interpolate everything to sync_topic timestamps. """
    
    d = defaultdict(lambda : [])

    for sync_ts in unpacked_bag[sync_topic][0]:

        for topic, (topic_ts, topic_msgs) in unpacked_bag.items():
            i_topic = np.argmin(np.abs(sync_ts - topic_ts))
            d[topic].append(topic_msgs[i_topic])

    for k,v in d.items():
        d[k] = np.stack(v)

    return d

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate dataset from bag files.')
    parser.add_argument('--bag_path', help="Path to bag directory.", required=True)
    parser.add_argument('--radar_buffer_len', type=int, default=2, help="Length of radar buffer.")
    parser.add_argument('--angle_res', type=int, default=1, help="Angle resolution for doppler-angle")
    parser.add_argument('--angle_range', type=int,default=90, help="Angle range for doppler-angle")
    parser.add_argument('--range_subsampling_factor', type=int, default=2, help="Range subsampling factor.")
    parser.add_argument('--normalization_range', nargs=2, type=float, default=[10.0, 25.0], help="Min max value for normalization")
    parser.add_argument('--resize_shape', nargs=2, type=int, default=[48, 48], help="Range subsampling factor.")
    parser.add_argument('--sync_topic', default='radar1', help="Sync topic..")
    args = parser.parse_args()

    print(f"Processing {args.bag_path}...")

    # Open bag file.
    bag = rosbag.Bag(args.bag_path)

    # Extract bag file.
    unpacked_bag = unpack_bag(bag,
                              radar_buffer_len=args.radar_buffer_len,
                              angle_res=args.angle_res, 
                              angle_range=args.angle_range,
                              range_subsampling_factor=args.range_subsampling_factor,
                              normalization_range=args.normalization_range,
                              resize_shape=args.resize_shape)

    print([*unpacked_bag.keys()])

    # Synchronize to radar timestamps.
    synced_bag = sync2topic(unpacked_bag, args.sync_topic)

    for k, v in synced_bag.items():
        print(f"{k}: {v.shape} {v.dtype}")

    # Save to .npz
    np.savez(os.path.splitext(args.bag_path)[0] + '.npz', **synced_bag) 

