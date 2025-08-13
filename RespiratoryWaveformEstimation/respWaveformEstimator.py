import numpy as np

from computeBreathingAngle import *
from computeMeanAccel import *
from filter import *
from findReferenceAxis import *
from trackRotationAxis import *
from computeRotationAxes import *
from computeRespWaveform import *

class RespWaveformEstimator:

    def __init__(self, sampling_freq=12.5, window_size=10, angle_threshold=5e-3):
        self.sampling_freq = sampling_freq
        self.window_size = window_size
        self.angle_threshold = angle_threshold

        self.accel

    def estimateRespWaveform(self, accel_data):

        # Low-pass filter
        filtered_data = filter_accelerometer_data(accel_data, cutoff=2.0, fs=self.sampling_freq, order=2)

        # Detect large motions
        static_mask = detect_large_motions(filtered_data, angle_threshold=self.angle_threshold)

        # doing this (keeping only static data) is wrong because it could cause the time series to be discontinuous!!
        # TODO: REVISE
        # on second thought, this is fine because we are only interested in the static data eh idk

        # Keep only static data
        static_accel_data = filtered_data[static_mask]

        # 'static_accel_data' should already be normalized if you want
        for t in range(len(static_accel_data)):
            static_accel_data[t] /= np.linalg.norm(static_accel_data[t])
        
        # Track the rotation axis
        smoothed_axes = track_rotation_axis(static_accel_data, window_size=self.window_size, angle_weight=True)

        # Compute āₜ (the mean acceleration over a window)
        mean_acc = compute_mean_accel(accel_data, window_size=5)

        # Make sure everything is the same length
        min_len = min(accel_data.shape[0], mean_acc.shape[0], smoothed_axes.shape[0])
        accel_data = accel_data[:min_len]
        mean_acc   = mean_acc[:min_len]
        smoothed_axes = smoothed_axes[:min_len]

        # Compute breathing angle (φₜ)
        phi = compute_breathing_angle(accel_data, mean_acc, smoothed_axes)

        # Derive the respiratory waveform
        resp_wave = compute_resp_waveform(phi, fs=self.sampling_freq, do_smooth=True, cutoff=1.0)

        return(resp_wave)
            