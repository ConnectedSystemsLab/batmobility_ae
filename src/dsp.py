#!/usr/bin/env python3

"""This file contains helper functions for handling radar data.
"""

import numpy as np
from numba import njit

""" Functions to convert from RadarFrame messages to radar cubes.. """

def reshape_frame(frame, flip_ods_phase=False, flip_aop_phase=False):
    """ Use this to reshape RadarFrameFull messages into radar cube.


    Args:
        frame (RadarFrameFull): RadarFrameFull message.
        flip_ods_phase (bool): Flip the phase of RX2 and RX3.
        flip_aop_phase (bool): Flip the phase of RX1 and RX3.

    Returns:
        radar_cube (np.ndarray): Radar cube. 
    """

    platform = frame.platform
    adc_output_fmt = frame.adc_output_fmt
    rx_phase_bias = np.array([a + 1j*b for a,b in zip(frame.rx_phase_bias[0::2],
                                                      frame.rx_phase_bias[1::2])])

    n_chirps  = int(frame.shape[0])
    rx        = np.array([int(x) for x in frame.rx])
    n_rx      = int(frame.shape[1])
    tx        = np.array([int(x) for x in frame.tx])
    n_tx      = int(sum(frame.tx))
    n_samples = int(frame.shape[2])

    return _reshape_frame(np.array(frame.data),
                          platform, adc_output_fmt, rx_phase_bias,
                          n_chirps, rx, n_rx, tx, n_tx, n_samples,
                          flip_ods_phase=flip_ods_phase,
                          flip_aop_phase=flip_aop_phase)

@njit(cache=True)
def _reshape_frame(data,
                   platform, adc_output_fmt, rx_phase_bias,
                   n_chirps, rx, n_rx, tx, n_tx, n_samples,
                   flip_ods_phase=False,
                   flip_aop_phase=False):
    """ Helper function for reshape_frame. 
        Refer to https://www.ti.com/lit/ug/swru546e/swru546e.pdf for details.
    """
    if adc_output_fmt > 0:

        radar_cube = np.zeros(len(data) // 2, dtype=np.complex64)

        radar_cube[0::2] = 1j*data[0::4] + data[2::4]
        radar_cube[1::2] = 1j*data[1::4] + data[3::4]

        radar_cube = radar_cube.reshape((n_chirps, 
                                         n_rx, 
                                         n_samples))

        # Apply RX phase correction for each antenna. 
        if 'xWR68xx' in platform:
            if flip_ods_phase: # Apply 180 deg phase change on RX2 and RX3
                c = 0
                for i_rx, rx_on in enumerate(rx):
                    if rx_on:
                        if i_rx == 1 or i_rx == 2:
                            radar_cube[:,c,:] *= -1
                        c += 1
            elif flip_aop_phase: # Apply 180 deg phase change on RX1 and RX3
                c = 0
                for i_rx, rx_on in enumerate(rx):
                    if rx_on:
                        if i_rx == 0 or i_rx == 2:
                            radar_cube[:,c,:] *= -1
                        c += 1


        radar_cube = radar_cube.reshape((n_chirps//n_tx, 
                                         n_rx*n_tx, 
                                         n_samples))

        # Apply RX phase correction from calibration.
        c = 0
        for i_tx, tx_on in enumerate(tx):
            if tx_on:
                for i_rx, rx_on in enumerate(rx):
                    if rx_on:
                        v_rx = i_tx*len(rx) + i_rx
                        # print(v_rx)
                        radar_cube[:,c,:] *= rx_phase_bias[v_rx]
                        c += 1

    else:
        radar_cube = data.reshape((n_chirps//n_tx, 
                                   n_rx*n_tx, 
                                   n_samples)).astype(np.complex64)

    return radar_cube

def reshape_frame_tdm(frame, flip_ods_phase=False, flip_aop_phase=False):
    """ Use this to reshape RadarFrameFull messages if using TDM.

    Args:
        frame (RadarFrameFull): RadarFrameFull message.
        flip_ods_phase (bool): Flip the phase of RX2 and RX3.
        flip_aop_phase (bool): Flip the phase of RX1 and RX3.

    Returns:
        radar_cube (np.ndarray): Radar cube.
    """

    platform = frame.platform
    adc_output_fmt = frame.adc_output_fmt
    rx_phase_bias = np.array([a + 1j*b for a,b in zip(frame.rx_phase_bias[0::2],
                                                      frame.rx_phase_bias[1::2])])

    n_chirps  = int(frame.shape[0])
    rx        = np.array([int(x) for x in frame.rx])
    n_rx      = int(frame.shape[1])
    tx        = np.array([int(x) for x in frame.tx])
    n_tx      = int(sum(frame.tx))
    n_samples = int(frame.shape[2])

    return _reshape_frame_tdm(np.array(frame.data),
                              platform, adc_output_fmt, rx_phase_bias,
                              n_chirps, rx, n_rx, tx, n_tx, n_samples,
                              flip_ods_phase=flip_ods_phase,
                              flip_aop_phase=flip_aop_phase)

@njit(cache=True)
def _tdm(radar_cube, n_tx, n_rx):
    """ Implements "Compensation of Motion-Induced Phase Errors in TDM MIMO Radars
        https://d-nb.info/1161008624/34
    """
    radar_cube_tdm = np.zeros((radar_cube.shape[0]*n_tx, 
                               radar_cube.shape[1], 
                               radar_cube.shape[2]), 
                               dtype=np.complex64)

    for i in range(n_tx):
        radar_cube_tdm[i::n_tx,i*n_rx:(i+1)*n_rx] \
                = radar_cube[:,i*n_rx:(i+1)*n_rx]

    return radar_cube_tdm

@njit(cache=True)
def _reshape_frame_tdm(data,
                       platform, adc_output_fmt, rx_phase_bias,
                       n_chirps, rx, n_rx, tx, n_tx, n_samples,
                       flip_ods_phase=False,
                       flip_aop_phase=False):
    """ Helper function for reshape_frame_tdm. 
        Refer to https://www.ti.com/lit/ug/swru546e/swru546e.pdf for details.
    """

    radar_cube = _reshape_frame(data, 
                                platform, adc_output_fmt, rx_phase_bias,
                                n_chirps, rx, n_rx, tx, n_tx, n_samples,
                                flip_ods_phase=flip_ods_phase,
                                flip_aop_phase=flip_aop_phase)

    radar_cube_tdm = _tdm(radar_cube, n_tx, n_rx)

    return radar_cube_tdm


""" AoA estimation functions. Inspired by https://github.com/PreSenseRadar/OpenRadar/ """

@njit(cache=True)
def get_mean(x, axis=0):
    """ Calculates the mean of a given set of input data (x=inputData).
    """
    return np.sum(x, axis=axis)/x.shape[axis]

@njit(cache=True)
def cov_matrix(x):
    """ Calculates the spatial covariance matrix (Rxx) for a given set of input data (x=inputData).
        Assumes rows denote Vrx axis.

    Args:
        x (ndarray): A 2D-Array with shape (rx, adc_samples) slice of the output of the 1D range fft

    Returns:
        Rxx (ndarray): A 2D-Array with shape (rx, rx)
    """

    _, num_adc_samples = x.shape
    x_T = x.T
    Rxx = x @ np.conjugate(x_T)
    Rxx = np.divide(Rxx, num_adc_samples)

    return Rxx 

@njit(cache=True)
def generate_steering_vectors(ang_est_range, ang_est_resolution, num_ant):
    """Generate steering vectors for AOA estimation given the theta range, theta resolution, and number of antennas

    Args:
        ang_est_range (int): The desired span of thetas for the angle spectrum.
        ang_est_resolution (float): The desired resolution in terms of theta
        num_ant (int): The number of Vrx antenna signals captured in the RDC

    Returns:
        Tuple[int, ndarray]: A tuple containing the number of vectors generated (integer divide angEstRange/ang_est_resolution)
        and the generated 2D-array steering vector of size (num_vec,num_ant)

    """
    num_vec = int(round((2 * ang_est_range / ang_est_resolution) + 1))
    steering_vectors = np.zeros((num_vec, num_ant), dtype='complex64')
    for kk in range(num_vec):
        for jj in range(num_ant):
            mag = -1 * np.pi * jj * np.sin((-ang_est_range + kk * ang_est_resolution) * np.pi / 180)
            real = np.cos(mag)
            imag = np.sin(mag)

            steering_vectors[kk, jj] = np.complex(real, imag)

    return steering_vectors

@njit(cache=True)
def aoa_bartlett(steering_vectors, signal_input):
    """Perform AOA estimation using Bartlett Beamforming

    Args:
        steering_vectors (ndarray): A 2D-array of size (num_theta, num_ant) generated from generate_steering_vectors
        signal_input (ndarray): Either a 2D-array or 3D-array of size (num_ant, num_chirps) or (num_chirps, num_vrx, num_adc_samples) respectively, containing ADC sample data sliced as described

    Returns:
        doa_spectrum (ndarray): A 3D-array of size (num_chirps, num_theta, num_range)

    """
    num_theta = steering_vectors.shape[0]
    num_rx = signal_input.shape[1]
    num_range = signal_input.shape[2]
    doa_spectrum = np.zeros((signal_input.shape[0], num_theta, num_range), dtype='complex64')
    for i in range(signal_input.shape[0]):
        doa_spectrum[i] = np.conjugate(steering_vectors) @ signal_input[i]
    return doa_spectrum

""" Radar cube processing functions. """

def compute_altitude(radar_cube, range_res, range_bias, window_len=3):
    """Estimate altitude based on the range response in the boresight direction.

    Args:
        radar_cube (ndarray): Input radar data cube of shape (num_chirps, num_rx, num_range_bins).
        range_res (float): Range resolution in meters.
        range_bias (float): Range bias in meters.
        window_len (int, optional): Length of the window for smoothing the range response. Default is 3.

    Returns:
        float: Estimated altitude in meters.

    """

    radar_cube_ = radar_cube - np.mean(radar_cube, axis=0)
    sum_rx = np.sum(radar_cube_, axis=1)

    range_response = np.fft.fft(sum_rx, axis=1)
    range_response_1d = np.sum(np.abs(range_response), axis=0)
    range_response_1d_ = np.zeros(len(range_response_1d) - window_len + 1)
    for r in range(len(range_response_1d_)):
        range_response_1d_[r] = np.sum(range_response_1d[r:r + window_len])

    range_bin = np.argmax(range_response_1d_)

    altitude = max(range_res * range_bin + range_bias, 0.0)

    return altitude

def compute_doppler_angle(radar_cube,
                          angle_res=1,
                          angle_range=90,
                          range_initial_bin=0,
                          range_subsampling_factor=2):
    """Computes the doppler-angle response.

    Args:
        radar_cube (ndarray): Input radar data cube of shape (num_chirps, num_rx, num_samples).
        angle_res (float, optional): Angular resolution in degrees. Default is 1.
        angle_range (float, optional): Angular range in degrees. Default is 90.
        range_initial_bin (int, optional): Initial range bin index for subsampling. Default is 0.
        range_subsampling_factor (int, optional): Range subsampling factor. Default is 2.

    Returns:
        ndarray: The computed doppler-angle heatmap.

    """

    n_chirps = radar_cube.shape[0]
    n_rx = radar_cube.shape[1]
    n_samples = radar_cube.shape[2]
    n_angle_bins = (angle_range * 2) // angle_res + 1

    # Subsample range bins.
    radar_cube_ = radar_cube[:, :, range_initial_bin::range_subsampling_factor]
    radar_cube_ -= np.mean(radar_cube_, axis=0)

    # Doppler processing.
    doppler_cube = np.fft.fft(radar_cube_, axis=0)
    doppler_cube = np.fft.fftshift(doppler_cube, axes=0)
    doppler_cube = np.asarray(doppler_cube, dtype=np.complex64)

    # Azimuth processing.
    steering_vectors = generate_steering_vectors(angle_range, angle_res, n_rx)

    doppler_angle_cube = aoa_bartlett(steering_vectors, doppler_cube)
    doppler_angle_cube -= np.expand_dims(np.mean(doppler_angle_cube, axis=2), axis=2)

    doppler_angle = np.log(np.mean(np.abs(doppler_angle_cube) ** 2, axis=2))

    return doppler_angle

""" Helper functions. """

def normalize(data, min_val=None, max_val=None):
    """ Normalize floats to [0.0, 1.0].
    """
    if min_val is None:
        min_val = np.min(data)
    if max_val is None:
        max_val = np.max(data)
    img = (((data-min_val)/(max_val-min_val)).clip(0.0, 1.0)).astype(data.dtype)
    return img

